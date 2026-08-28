"""
Door, doorway and exit detection.

WHY THIS IS NOT YOLO

The object detector is trained on COCO, which has no door class. Pointed at a
corridor with three doors it returns nothing at all — verified. Doors have to
come from somewhere else, so they come from Gemini vision, which was measured
on a corridor fixture with known ground truth:

    gemini-3.1-flash-lite   3.4s   3/3 doors, correct sides, exit correctly
                                   identified from the EXIT sign above it
    gemini-3.6-flash       45.6s   3/3 including "ajar", but far too slow to
                                   be useful to someone walking

flash-lite is used. Its one measured weakness is calling a slightly-ajar door
"closed"; open versus closed is reliable, and that is the distinction that
matters for walking through one.

On a football-pitch photograph it reported zero doors, so it is not inventing
them to satisfy the question.

COST CONTROL

At ~3.4s and a per-call charge this cannot run per frame. It runs on demand,
and on a slow background sweep while the user is moving. Everything else in
the perception stack stays local.

WHAT IT REFUSES TO CLAIM

An exit is only an exit with visible evidence — an EXIT sign, or daylight
through the opening. "This door probably leads outside" is exactly the kind of
guess that walks a blind user into a fire escape or a stairwell.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from app.perception.direction import DirectionBucket, _raw_bucket

logger = logging.getLogger("vidi.perception.doors")

DOOR_MODEL = "gemini-3.1-flash-lite"

# Below this we do not mention the door at all.
MIN_DOOR_CONFIDENCE = 0.55
# Below this we mention it but hedge.
CONFIDENT_DOOR = 0.80

# How close two observations must be (normalized centre distance) to count as
# the same physical door between sweeps.
SAME_DOOR_DISTANCE = 0.18
# A door unseen for this long is dropped.
FORGET_AFTER_S = 25.0


class DoorState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    AJAR = "ajar"
    UNKNOWN = "unknown"


DOOR_PROMPT = """You are the navigation system for a blind user's assistive device.

Find every DOOR, DOORWAY, GATE, ELEVATOR or STAIRWELL ENTRANCE in this image.

For each one report:
- kind: door, doorway, gate, elevator, stairs, or exit
- state: open, closed, ajar, or unknown (use unknown if you genuinely cannot tell)
- is_exit: true ONLY if there is visible evidence it leads out - an EXIT sign,
  or daylight/outdoors visible through it. Do NOT infer it from position.
- evidence: the specific visible thing supporting state and is_exit
- box_2d: [ymin, xmin, ymax, xmax] normalized 0-1000
- confidence: 0.0-1.0

CRITICAL RULES:
- Report only what is actually visible. If there are no doors, return an empty
  list. Never invent one to be helpful.
- Never claim a door is an exit without visible evidence. Sending a blind user
  through the wrong door - a fire escape, a stairwell, a service door - is
  dangerous.
- If you cannot tell whether it is open, say unknown rather than guessing."""

DOOR_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "doors": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "kind": {"type": "STRING"},
                    "state": {"type": "STRING"},
                    "is_exit": {"type": "BOOLEAN"},
                    "evidence": {"type": "STRING"},
                    "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["kind", "state", "is_exit", "evidence",
                             "box_2d", "confidence"],
            },
        }
    },
    "required": ["doors"],
}


@dataclass
class Door:
    kind: str
    state: DoorState
    is_exit: bool
    evidence: str
    confidence: float
    direction: DirectionBucket
    centre_x: float                      # normalized 0..1
    height_fraction: float               # rough proximity cue
    first_seen: float = 0.0
    last_seen: float = 0.0
    announced: bool = False

    @property
    def key(self) -> str:
        return "door:%.2f" % round(self.centre_x, 2)

    @property
    def is_confident(self) -> bool:
        return self.confidence >= CONFIDENT_DOOR

    def direction_phrase(self) -> str:
        from app.agent.tools import DIRECTION_PHRASE

        return DIRECTION_PHRASE.get(self.direction.value, self.direction.value)

    def describe(self) -> str:
        """Spoken form, hedged when the read is weak."""
        noun = "an exit" if self.is_exit else ("a doorway" if self.kind == "doorway"
                                              else "a " + self.kind)
        opener = "There's " if self.is_confident else "There may be "
        sentence = opener + noun + " " + self.direction_phrase()

        if self.state == DoorState.OPEN:
            sentence += ", and it's open"
        elif self.state == DoorState.CLOSED:
            sentence += ", and it's closed"
        elif self.state == DoorState.AJAR:
            sentence += ", slightly open"
        return sentence + "."

    def payload(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "state": self.state.value,
            "is_exit": self.is_exit,
            "evidence": self.evidence,
            "confidence": round(self.confidence, 2),
            "is_confident": self.is_confident,
            "direction": self.direction.value,
            "direction_phrase": self.direction_phrase(),
            "spoken": self.describe(),
            "seconds_since_seen": round(time.time() - self.last_seen, 1),
        }


def _direction_from_box(box: Optional[List[int]]) -> tuple:
    if not box or len(box) != 4:
        return DirectionBucket.CENTER, 0.5, 0.0
    try:
        centre = (float(box[1]) + float(box[3])) / 2 / 1000.0
        height = abs(float(box[2]) - float(box[0])) / 1000.0
    except (TypeError, ValueError):
        return DirectionBucket.CENTER, 0.5, 0.0
    centre = min(max(centre, 0.0), 1.0)
    return _raw_bucket(centre), centre, height


class DoorDetector:
    """Gemini-backed. Never raises: no doors is a valid, common answer."""

    def __init__(self, api_key: Optional[str], model: str = DOOR_MODEL) -> None:
        self._api_key = api_key
        self.model = model
        self.last_latency_ms: Optional[float] = None
        self.calls = 0

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    async def detect(self, frame: np.ndarray) -> List[Door]:
        if not self.available or frame is None:
            return []

        import httpx

        from app.agent.vision_llm import API_BASE, encode_frame

        try:
            encoded = encode_frame(frame)
        except ValueError:
            return []

        body = {
            "contents": [{"parts": [
                {"text": DOOR_PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": encoded}},
            ]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": DOOR_SCHEMA,
                # Deterministic: the same doorway must not change state
                # between sweeps just from sampling noise.
                "temperature": 0.0,
            },
        }

        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    API_BASE + self.model + ":generateContent",
                    headers={"x-goog-api-key": self._api_key},
                    json=body,
                )
            self.last_latency_ms = (time.monotonic() - started) * 1000
            self.calls += 1
            if response.status_code != 200:
                logger.warning("door detection failed: %s %s",
                               response.status_code, response.text[:160])
                return []
            import json as _json

            raw = _json.loads(
                response.json()["candidates"][0]["content"]["parts"][0]["text"]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("door detection errored: %s", str(exc)[:160])
            return []

        now = time.time()
        doors: List[Door] = []
        for item in raw.get("doors", []):
            confidence = float(item.get("confidence", 0.0))
            if confidence < MIN_DOOR_CONFIDENCE:
                continue
            direction, centre, height = _direction_from_box(item.get("box_2d"))
            try:
                state = DoorState(str(item.get("state", "unknown")).lower())
            except ValueError:
                state = DoorState.UNKNOWN
            doors.append(Door(
                kind=str(item.get("kind", "door")).lower(),
                state=state,
                is_exit=bool(item.get("is_exit")),
                evidence=str(item.get("evidence", ""))[:160],
                confidence=confidence,
                direction=direction,
                centre_x=centre,
                height_fraction=height,
                first_seen=now,
                last_seen=now,
            ))
        doors.sort(key=lambda d: d.centre_x)
        return doors


def scene_changed(previous, current, threshold: float = 6.0) -> bool:
    """
    Has the view moved enough to be worth paying for another door sweep?

    Each sweep is a billed Gemini call. Someone sitting still would otherwise
    buy the same answer every few seconds forever. A cheap mean-absolute
    difference over a downscaled greyscale pair costs well under a millisecond
    and skips the vast majority of redundant sweeps.

    Deliberately a LOW threshold: missing a door because we were too thrifty is
    far worse than an occasional wasted call.
    """
    if previous is None or current is None:
        return True
    try:
        import cv2

        a = cv2.cvtColor(cv2.resize(previous, (96, 96)), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(cv2.resize(current, (96, 96)), cv2.COLOR_BGR2GRAY)
        return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))) >= threshold
    except Exception:  # noqa: BLE001 - if in doubt, sweep
        return True


class DoorTracker:
    """
    Remembers doors between sweeps so "the door just opened" is reportable.

    Matching is by horizontal position, which is crude but appropriate: sweeps
    are seconds apart and a door does not move. Two doors closer together than
    SAME_DOOR_DISTANCE would be conflated, and that is an accepted limit.
    """

    def __init__(self) -> None:
        self.doors: Dict[str, Door] = {}

    def update(self, observed: Sequence[Door], now: Optional[float] = None) -> List[tuple]:
        """Feed one sweep. Returns (EventType, Door, spoken) worth reporting."""
        from app.events.bus import EventType

        now = now if now is not None else time.time()
        events: List[tuple] = []
        matched_keys = set()

        for door in observed:
            existing = self._match(door)
            if existing is None:
                door.first_seen = door.last_seen = now
                self.doors[door.key] = door
                matched_keys.add(door.key)
                if door.is_exit:
                    events.append((EventType.PLACE_DETECTED, door,
                                   "I can see an exit " + door.direction_phrase() + "."))
                else:
                    events.append((EventType.OBJECT_DETECTED, door, door.describe()))
                continue

            matched_keys.add(existing.key)
            previous_state = existing.state
            existing.state = door.state
            existing.confidence = max(existing.confidence, door.confidence)
            existing.is_exit = existing.is_exit or door.is_exit
            existing.direction = door.direction
            existing.centre_x = door.centre_x
            existing.evidence = door.evidence or existing.evidence
            existing.last_seen = now

            # THE transition the user asked for.
            if previous_state != door.state and DoorState.UNKNOWN not in (
                previous_state, door.state
            ):
                if door.state == DoorState.OPEN:
                    events.append((EventType.OBJECT_DETECTED, existing,
                                   "The door " + existing.direction_phrase()
                                   + " has opened."))
                elif door.state == DoorState.CLOSED:
                    events.append((EventType.OBJECT_DETECTED, existing,
                                   "The door " + existing.direction_phrase()
                                   + " has closed."))

        self._expire(now, matched_keys)
        return events

    def _match(self, door: Door) -> Optional[Door]:
        best, best_gap = None, SAME_DOOR_DISTANCE
        for existing in self.doors.values():
            gap = abs(existing.centre_x - door.centre_x)
            if gap < best_gap:
                best, best_gap = existing, gap
        return best

    def _expire(self, now: float, seen_keys: set) -> None:
        for key in [k for k, d in self.doors.items()
                    if k not in seen_keys and now - d.last_seen > FORGET_AFTER_S]:
            del self.doors[key]

    def current(self, max_age_s: float = FORGET_AFTER_S,
                now: Optional[float] = None) -> List[Door]:
        # now is injectable so callers driving a synthetic clock (tests, replay)
        # do not have every door instantly judged stale against wall time.
        now = now if now is not None else time.time()
        live = [d for d in self.doors.values() if now - d.last_seen <= max_age_s]
        # Exits first, then confident doors, then by position.
        live.sort(key=lambda d: (not d.is_exit, -d.confidence, d.centre_x))
        return live

    def exits(self, now: Optional[float] = None) -> List[Door]:
        return [d for d in self.current(now=now) if d.is_exit]

    def reset(self) -> None:
        self.doors.clear()
