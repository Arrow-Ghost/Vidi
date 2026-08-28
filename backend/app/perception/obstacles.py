"""
Proximity monitoring — "there's a chair right in front of you".

This is the only part of the system allowed to interrupt unconditionally, so
it is also the part most able to ruin the experience. Two failure modes, and
they pull in opposite directions:

  Too quiet  the user walks into a chair.
  Too loud   the user turns the assistant off, and then walks into a chair.

The rules below exist to sit between those.

WHAT COUNTS AS WORTH INTERRUPTING FOR

  1. It is CLOSE. Distance comes from the fused depth estimator, and if that
     estimator is not confident we do not raise an alarm from a guess.
  2. It is IN THE WAY. Something close but far to the left is not on a
     collision course; only the central corridor of the frame counts.
  3. We are SURE it is there. A 0.4-confidence detection is not worth a
     hazard interrupt.
  4. It is NEW, or it has got meaningfully closer. Re-announcing the same
     chair every two seconds while the user stands still is the fastest way
     to make someone stop trusting this.

WHAT THIS NEVER SAYS

It never reports that the way ahead is clear. Detecting nothing means the
camera saw nothing it recognises — glass doors, steps down, low bollards,
anything outside the detector's vocabulary, and anything outside its narrow
field of view are all invisible to it. Saying "clear" would be a promise this
system cannot keep, and a blind user acting on it could be badly hurt.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from app.perception.direction import DirectionBucket
from app.perception.distance import DistanceRange

logger = logging.getLogger("vidi.perception.obstacles")

# The central corridor: what the user would actually walk into.
PATH_DIRECTIONS = {
    DirectionBucket.CENTER_LEFT,
    DirectionBucket.CENTER,
    DirectionBucket.CENTER_RIGHT,
}
# Immediately ahead — a stricter band used to escalate severity.
DIRECTLY_AHEAD = {DirectionBucket.CENTER}

# Ranges near enough to matter, most urgent first.
URGENT_RANGES = {DistanceRange.VERY_CLOSE}
NEAR_RANGES = {DistanceRange.VERY_CLOSE, DistanceRange.CLOSE}

# Below these we do not raise a hazard, because we are not sure enough.
MIN_DETECTION_CONFIDENCE = 0.55
MIN_DISTANCE_CONFIDENCE = 0.40

# Per-object gap. Long enough not to nag, short enough to re-warn while
# actually approaching something.
REPEAT_COOLDOWN_S = 12.0
# A closing step (nearby -> close -> very close) bypasses the cooldown once,
# because getting closer is genuinely new information.
ESCALATION_BYPASSES_COOLDOWN = True

# Things worth naming as obstacles. A detected banana is not a hazard.
HAZARD_CLASSES = {
    "person", "chair", "dining table", "bench", "couch", "bed", "tv",
    "potted plant", "bicycle", "motorcycle", "car", "bus", "truck",
    "suitcase", "backpack", "refrigerator", "oven", "toilet", "sink",
}

# Spoken names where the detector's label is not what a person would say.
SPOKEN_NAME = {
    "dining table": "a table",
    "potted plant": "a large plant",
    "tv": "a screen",
    "couch": "a sofa",
}

_RANGE_ORDER = {
    DistanceRange.VERY_CLOSE: 0,
    DistanceRange.CLOSE: 1,
    DistanceRange.NEARBY: 2,
    DistanceRange.FAR: 3,
    DistanceRange.UNKNOWN: 4,
}


@dataclass
class Obstacle:
    entity_id: str
    type: str
    direction: DirectionBucket
    distance_range: DistanceRange
    distance_phrase: str
    detection_confidence: float
    distance_confidence: float
    is_person: bool
    urgent: bool

    @property
    def spoken_name(self) -> str:
        if self.is_person:
            return "someone"
        return SPOKEN_NAME.get(self.type, "a " + self.type)

    def describe(self) -> str:
        """
        One short sentence. States position and proximity, promises nothing
        about anything it did not see.
        """
        where = {
            DirectionBucket.CENTER: "directly ahead of you",
            DirectionBucket.CENTER_LEFT: "just ahead, slightly to your left",
            DirectionBucket.CENTER_RIGHT: "just ahead, slightly to your right",
        }.get(self.direction, "ahead of you")

        if self.urgent:
            return self.spoken_name.capitalize() + " " + where + ", very close."
        return self.spoken_name.capitalize() + " " + where + ", " + self.distance_phrase + "."

    def payload(self) -> Dict:
        return {
            "entity_id": self.entity_id,
            "type": self.type,
            "spoken_name": self.spoken_name,
            "direction": self.direction.value,
            "distance_range": self.distance_range.value,
            "distance_phrase": self.distance_phrase,
            "detection_confidence": round(self.detection_confidence, 2),
            "distance_confidence": round(self.distance_confidence, 2),
            "urgent": self.urgent,
            "is_person": self.is_person,
            "spoken": self.describe(),
        }


@dataclass
class _Seen:
    announced_at: float
    range_order: int


class ObstacleMonitor:
    def __init__(self, clock=time.time) -> None:
        self._clock = clock
        self._seen: Dict[str, _Seen] = {}
        self.alerts_raised = 0
        self.suppressed = 0

    # ------------------------------------------------------------------

    def scan(self, entities: Sequence) -> List[Obstacle]:
        """
        Everything currently close enough and central enough to matter,
        regardless of whether it has already been announced. This is what the
        `check_path` tool reports when the user actually asks.
        """
        found: List[Obstacle] = []
        for entity in entities:
            obstacle = self._as_obstacle(entity)
            if obstacle is not None:
                found.append(obstacle)

        found.sort(key=lambda o: (
            _RANGE_ORDER[o.distance_range],
            0 if o.direction in DIRECTLY_AHEAD else 1,
            -o.detection_confidence,
        ))
        return found

    def poll(self, entities: Sequence) -> List[Obstacle]:
        """
        Only what is worth INTERRUPTING for right now: new hazards, or ones
        that have moved into a closer band since we last spoke about them.
        """
        now = self._clock()
        worth_saying: List[Obstacle] = []

        for obstacle in self.scan(entities):
            order = _RANGE_ORDER[obstacle.distance_range]
            previous = self._seen.get(obstacle.entity_id)

            if previous is None:
                worth_saying.append(obstacle)
            elif order < previous.range_order and ESCALATION_BYPASSES_COOLDOWN:
                # Genuinely closer than when we last mentioned it.
                worth_saying.append(obstacle)
            elif now - previous.announced_at >= REPEAT_COOLDOWN_S:
                worth_saying.append(obstacle)
            else:
                self.suppressed += 1
                continue

            self._seen[obstacle.entity_id] = _Seen(now, order)

        self._forget_stale(entities, now)
        self.alerts_raised += len(worth_saying)
        return worth_saying

    # ------------------------------------------------------------------

    def _as_obstacle(self, entity) -> Optional[Obstacle]:
        entity_type = getattr(entity, "type", "")
        if entity_type not in HAZARD_CLASSES:
            return None

        direction = entity.direction.bucket
        if direction not in PATH_DIRECTIONS:
            return None            # beside the user, not in their way

        distance = entity.distance
        if distance.range not in NEAR_RANGES:
            return None            # far enough away to be ordinary scenery

        if entity.confidence < MIN_DETECTION_CONFIDENCE:
            return None            # not sure enough to raise an alarm
        if distance.confidence < MIN_DISTANCE_CONFIDENCE:
            # We think it is close but the depth estimate is weak. Not a
            # hazard interrupt — describe_scene still reports it if asked.
            return None

        urgent = (
            distance.range in URGENT_RANGES
            and direction in DIRECTLY_AHEAD
        )
        return Obstacle(
            entity_id=entity.id,
            type=entity_type,
            direction=direction,
            distance_range=distance.range,
            distance_phrase=distance.label,
            detection_confidence=entity.confidence,
            distance_confidence=distance.confidence,
            is_person=getattr(entity, "is_person", entity_type == "person"),
            urgent=urgent,
        )

    def _forget_stale(self, entities: Sequence, now: float) -> None:
        """Drop memory of things no longer tracked, so a returning object re-alerts."""
        live = {getattr(e, "id", None) for e in entities}
        for entity_id in [k for k in self._seen if k not in live]:
            del self._seen[entity_id]

    def stats(self) -> Dict:
        return {
            "alerts_raised": self.alerts_raised,
            "suppressed": self.suppressed,
            "tracked": len(self._seen),
        }

    def reset(self) -> None:
        self._seen.clear()
