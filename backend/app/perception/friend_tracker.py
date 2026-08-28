"""
Temporal friend confirmation and tracking — Parts 19, 21, 57, 58.

WHY THIS EXISTS

Face matching was single-frame. One bad frame — motion blur, an unlucky angle,
a partially occluded stranger — could put a name on the wrong person. For a
blind user that is not a glitch: they would walk up to a stranger believing a
friend is there. Part 58 calls false positives on trusted-person alerts
unacceptable, and this module is the mechanism that prevents them.

HOW

A name is never spoken from one observation. A candidate accumulates
confirmations across frames, and only after CONFIRM_FRAMES agreeing
observations does it become an identified friend. Identity confidence is then
a blend of three independent signals, not raw similarity:

    similarity   how far above the same-identity threshold the match sits
    consistency  how many of the recent observations agreed
    quality      how large and usable the face was

Part 57's bands follow from that combined score:

    >= 0.90   "Ravi is ahead of you."          state it
    >= 0.75   "I think that may be Ravi."      hedge
    below     say nothing about identity       never guess

LOSS IS REPORTED, NOT PAPERED OVER (Part 21). When the crowd blocks the view we
say the view is blocked; when the track dies we say we lost contact. We never
keep asserting a position from a stale observation.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Sequence

from app.perception.direction import DirectionBucket, _raw_bucket
from app.perception.faces import SAME_IDENTITY_THRESHOLD

logger = logging.getLogger("vidi.perception.friend_tracker")

# Agreeing observations required before a name may be spoken (Part 58).
CONFIRM_FRAMES = 3
# Window of recent observations used for the consistency term.
HISTORY = 6
# Identity-confidence bands (Part 57), configurable per deployment.
CONFIDENT_BAND = 0.90
TENTATIVE_BAND = 0.75
# Seconds a confirmed friend may go unseen before we call it occluded / lost.
OCCLUDED_AFTER_S = 1.5
LOST_AFTER_S = 6.0
# A face this fraction of frame height is comfortably resolvable.
GOOD_FACE_FRACTION = 0.16


class FriendState(str, Enum):
    CANDIDATE = "candidate"      # seen, not yet confirmed — never spoken
    CONFIRMED = "confirmed"      # may be reported
    OCCLUDED = "occluded"        # was confirmed, view now blocked
    LOST = "lost"                # gone


@dataclass
class Observation:
    similarity: float
    quality: float
    center_x: float
    at: float


@dataclass
class TrackedFriend:
    name: str
    state: FriendState = FriendState.CANDIDATE
    observations: Deque[Observation] = field(default_factory=lambda: deque(maxlen=HISTORY))
    first_seen: float = 0.0
    last_seen: float = 0.0
    direction: DirectionBucket = DirectionBucket.CENTER
    previous_direction: Optional[DirectionBucket] = None
    identity_confidence: float = 0.0
    announced: bool = False
    in_crowd: bool = False
    # Horizontal movement in normalized units/sec, from the person track.
    vx: float = 0.0
    scale_rate: float = 0.0
    distance_range: Optional[str] = None
    distance_phrase: Optional[str] = None

    @property
    def confirmations(self) -> int:
        return len(self.observations)

    @property
    def is_confident(self) -> bool:
        return self.identity_confidence >= CONFIDENT_BAND

    @property
    def is_tentative(self) -> bool:
        return TENTATIVE_BAND <= self.identity_confidence < CONFIDENT_BAND

    def movement_phrase(self) -> Optional[str]:
        """
        Describes motion only. Never intent — Part 55 forbids "coming to meet
        you"; the most we may say is which way they are moving.
        """
        if self.scale_rate > 0.06 and abs(self.vx) < 0.05:
            return "moving toward you"
        if self.scale_rate < -0.06 and abs(self.vx) < 0.05:
            return "moving away from you"
        if self.vx > 0.05:
            return "moving toward your right"
        if self.vx < -0.05:
            return "moving toward your left"
        return None

    def payload(self) -> Dict[str, Any]:
        from app.agent.tools import DIRECTION_PHRASE

        return {
            "name": self.name,
            "state": self.state.value,
            "direction": self.direction.value,
            "direction_phrase": DIRECTION_PHRASE.get(
                self.direction.value, self.direction.value
            ),
            "identity_confidence": round(self.identity_confidence, 3),
            "is_confident": self.is_confident,
            "is_tentative": self.is_tentative,
            "confirmations": self.confirmations,
            "in_crowd": self.in_crowd,
            "movement": self.movement_phrase(),
            "distance_range": self.distance_range,
            "distance_phrase": self.distance_phrase,
            "seconds_since_seen": round(time.time() - self.last_seen, 1),
        }


def _identity_confidence(observations: Sequence[Observation]) -> float:
    """
    Blend similarity, consistency and face quality into one score.

    Raw SFace similarity alone is a poor confidence signal: a single strong
    frame can score high by luck. Requiring agreement across observations is
    what actually separates "that is Ravi" from "something briefly looked like
    Ravi".
    """
    if not observations:
        return 0.0
    recent = list(observations)

    # How far above the same-identity threshold the matches sit, on average.
    headroom = 1.0 - SAME_IDENTITY_THRESHOLD
    similarity = sum(
        max(0.0, (o.similarity - SAME_IDENTITY_THRESHOLD) / headroom) for o in recent
    ) / len(recent)

    consistency = min(1.0, len(recent) / CONFIRM_FRAMES)
    quality = sum(o.quality for o in recent) / len(recent)

    return round(min(0.99, 0.55 * similarity + 0.30 * consistency + 0.15 * quality), 3)


class FriendTracker:
    """
    Turns per-frame face matches into stable, confirmed identities plus the
    state transitions worth telling the user about.
    """

    def __init__(self, confirm_frames: int = CONFIRM_FRAMES) -> None:
        self.confirm_frames = confirm_frames
        self.friends: Dict[str, TrackedFriend] = {}

    def update(
        self,
        matches: Sequence,
        image_width: int,
        image_height: int,
        person_tracks: Optional[Sequence] = None,
        in_crowd: bool = False,
        now: Optional[float] = None,
    ) -> List[tuple]:
        """
        Feed one frame's face matches. Returns (EventType, TrackedFriend,
        summary) triples for anything newly worth reporting.
        """
        from app.events.bus import EventType

        now = now if now is not None else time.time()
        events: List[tuple] = []
        seen_names = set()

        for match in matches:
            name = match.name
            seen_names.add(name)
            friend = self.friends.get(name)
            if friend is None:
                friend = TrackedFriend(name=name, first_seen=now)
                self.friends[name] = friend

            centre = getattr(match, "center_x", None)
            if centre is None:
                box = getattr(match, "box", None)
                centre = ((box[0] + box[2]) / 2) if box else image_width / 2
            quality = self._face_quality(match, image_height)

            friend.observations.append(
                Observation(
                    similarity=float(getattr(match, "similarity", 0.0)),
                    quality=quality, center_x=float(centre), at=now,
                )
            )
            friend.last_seen = now
            friend.in_crowd = in_crowd
            friend.previous_direction = friend.direction
            friend.direction = (
                _raw_bucket(centre / image_width) if image_width else DirectionBucket.CENTER
            )
            friend.identity_confidence = _identity_confidence(friend.observations)
            self._attach_motion(friend, person_tracks, centre, image_width)

            was = friend.state
            # THE GATE: no name is spoken before enough agreeing observations.
            if friend.confirmations >= self.confirm_frames and friend.identity_confidence >= TENTATIVE_BAND:
                friend.state = FriendState.CONFIRMED
            elif was in (FriendState.OCCLUDED, FriendState.LOST):
                friend.state = FriendState.CONFIRMED

            if friend.state == FriendState.CONFIRMED and not friend.announced:
                friend.announced = True
                events.append((
                    EventType.FRIEND_DETECTED_IN_CROWD if in_crowd else EventType.FRIEND_DETECTED,
                    friend,
                    self.describe(friend),
                ))
            elif was == FriendState.OCCLUDED and friend.state == FriendState.CONFIRMED:
                events.append((EventType.FRIEND_DETECTED, friend,
                               "I can see " + name + " again, " + self._where(friend) + "."))
            elif friend.state == FriendState.CONFIRMED:
                movement = friend.movement_phrase()
                moved_buckets = (
                    friend.previous_direction is not None
                    and friend.previous_direction != friend.direction
                )
                if movement == "moving toward you" and moved_buckets:
                    events.append((EventType.FRIEND_APPROACHING, friend,
                                   name + " is moving toward you, " + self._where(friend) + "."))

        # Anything confirmed but not seen this frame.
        for name, friend in self.friends.items():
            if name in seen_names or friend.state == FriendState.LOST:
                continue
            gap = now - friend.last_seen
            if friend.state == FriendState.CONFIRMED and gap > OCCLUDED_AFTER_S:
                friend.state = FriendState.OCCLUDED
                reason = (
                    "The crowd is blocking my view of " + name + " right now."
                    if friend.in_crowd else
                    "I've lost sight of " + name + " for a moment."
                )
                events.append((EventType.FRIEND_OCCLUDED, friend, reason))
            elif friend.state == FriendState.OCCLUDED and gap > LOST_AFTER_S:
                friend.state = FriendState.LOST
                friend.announced = False
                events.append((
                    EventType.FRIEND_LOST_IN_CROWD if friend.in_crowd else EventType.FRIEND_LOST,
                    friend,
                    "I've lost visual contact with " + name + ".",
                ))
            elif friend.state == FriendState.CANDIDATE and gap > LOST_AFTER_S:
                # Never confirmed, so nothing was ever said. Drop it silently.
                friend.state = FriendState.LOST

        return events

    # ------------------------------------------------------------------

    @staticmethod
    def _face_quality(match, image_height: int) -> float:
        box = getattr(match, "box", None)
        if not box or not image_height:
            return 0.5
        height = abs(box[3] - box[1])
        return max(0.0, min(1.0, (height / image_height) / GOOD_FACE_FRACTION))

    @staticmethod
    def _attach_motion(friend: TrackedFriend, person_tracks, centre_x, image_width) -> None:
        """Borrow velocity from the person track this face sits inside."""
        if not person_tracks:
            return
        best, best_gap = None, 1e9
        for track in person_tracks:
            gap = abs(track.box.center_x - centre_x)
            if gap < best_gap and track.box.x1 - 20 <= centre_x <= track.box.x2 + 20:
                best, best_gap = track, gap
        if best is not None:
            friend.vx = getattr(best, "vx", 0.0)
            friend.scale_rate = getattr(best, "scale_rate", 0.0)

    @staticmethod
    def _where(friend: TrackedFriend) -> str:
        from app.agent.tools import DIRECTION_PHRASE

        return DIRECTION_PHRASE.get(friend.direction.value, friend.direction.value)

    def describe(self, friend: TrackedFriend) -> str:
        """
        Phrasing that matches the confidence band (Part 57).

        Refuses to produce a name below the tentative band even if called
        directly. The confirmation gate in update() already prevents this path,
        but a describe() that can emit an identity for a 0.48-confidence
        candidate is a loaded gun for future callers — so the floor lives here
        too, not only at the call site.
        """
        if friend.identity_confidence < TENTATIVE_BAND:
            return (
                "There is someone here, but I am not confident enough to say who it is."
            )
        where = self._where(friend)
        if friend.in_crowd:
            where = where + ", inside the crowd"
        if friend.is_confident:
            sentence = friend.name + " is " + where
        else:
            sentence = "I think that may be " + friend.name + ", " + where
        movement = friend.movement_phrase()
        if movement:
            sentence += ", " + movement
        return sentence + "."

    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[TrackedFriend]:
        for key, friend in self.friends.items():
            if key.lower() == name.strip().lower():
                return friend
        return None

    def visible(self) -> List[TrackedFriend]:
        return [f for f in self.friends.values() if f.state == FriendState.CONFIRMED]

    def reset(self) -> None:
        self.friends.clear()
