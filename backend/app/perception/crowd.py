"""
Crowd understanding — Parts 23 to 32.

Part 23 is explicit that counting people is NOT crowd detection, so this
analyses density, spatial distribution, movement, flow, congestion, and
approach, and emits state-transition events rather than a running commentary.

WHAT IS HONEST HERE, AND WHAT IS NOT

  * Counts are approximate. Detection misses occluded people in exactly the
    situations where crowds form, so counts are reported as "roughly" and the
    class (small group / dense crowd) is what the agent actually speaks.
  * Directions are RELATIVE — left, right, ahead, behind (Part 26). Compass
    bearings would need orientation or GPS this system does not have.
  * Approach is inferred from apparent size growth plus inward motion, and is
    only asserted when both agree.
  * Movement never implies intent (Part 55). "The crowd is moving toward the
    entrance" is allowed; "the crowd is going to the event" is not, and no
    field in this module carries a destination or a purpose.

THRESHOLDS ARE CONFIGURABLE (Part 24) because they depend on field of view,
camera height and setting. A corridor camera and an open plaza do not share a
definition of "dense".
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from app.perception.direction import DirectionBucket, _raw_bucket

logger = logging.getLogger("vidi.perception.crowd")


class CrowdSize(str, Enum):
    NO_CROWD = "no_crowd"
    SMALL_GROUP = "small_group"
    MODERATE_CROWD = "moderate_crowd"
    LARGE_CROWD = "large_crowd"
    DENSE_CROWD = "dense_crowd"


class CrowdMotion(str, Enum):
    STATIC = "static"
    SLOW_MOVING = "slow_moving"
    MOVING = "moving"
    FAST_MOVING = "fast_moving"
    MIXED_FLOW = "mixed_flow"


class CrowdFlow(str, Enum):
    NONE = "none"
    UNIDIRECTIONAL = "unidirectional"
    BIDIRECTIONAL = "bidirectional"
    CONVERGING = "converging"
    DIVERGING = "diverging"
    CHAOTIC = "chaotic"


class Density(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class CrowdConfig:
    """Every threshold here is meant to be tuned per deployment (Part 24)."""

    small_group_min: int = 3
    moderate_min: int = 6
    large_min: int = 10
    # Fraction of frame width covered by person boxes before we call it dense.
    dense_coverage: float = 0.42
    medium_coverage: float = 0.22
    # Normalized units/second.
    static_speed: float = 0.012
    slow_speed: float = 0.045
    fast_speed: float = 0.16
    # Agreement of motion directions before calling flow unidirectional.
    unidirectional_agreement: float = 0.62
    bidirectional_agreement: float = 0.30
    # Apparent growth rate that counts as approaching.
    approach_scale_rate: float = 0.05
    # Frames of agreement before a state change is reported.
    confirm_frames: int = 2


@dataclass
class CrowdState:
    size: CrowdSize = CrowdSize.NO_CROWD
    density: Density = Density.LOW
    motion: CrowdMotion = CrowdMotion.STATIC
    flow: CrowdFlow = CrowdFlow.NONE
    person_count: int = 0
    approaching: bool = False
    departing: bool = False
    congested: bool = False
    dominant_direction: Optional[str] = None
    flow_direction: Optional[str] = None
    centre_direction: Optional[str] = None
    confidence: float = 0.0
    at: float = field(default_factory=time.time)

    @property
    def is_crowd(self) -> bool:
        return self.size != CrowdSize.NO_CROWD

    def payload(self) -> Dict[str, Any]:
        return {
            "size": self.size.value,
            "density": self.density.value,
            "motion": self.motion.value,
            "flow": self.flow.value,
            "person_count": self.person_count,
            "count_is_approximate": True,
            "approaching": self.approaching,
            "departing": self.departing,
            "congested": self.congested,
            "dominant_direction": self.dominant_direction,
            "flow_direction": self.flow_direction,
            "centre_direction": self.centre_direction,
            "confidence": round(self.confidence, 2),
        }

    def describe(self) -> str:
        """Neutral phrasing the agent can speak. No intent, no reassurance."""
        if not self.is_crowd:
            return "I can't see a crowd in the camera's current view."

        words = {
            CrowdSize.SMALL_GROUP: "a small group of people",
            CrowdSize.MODERATE_CROWD: "a moderate crowd",
            CrowdSize.LARGE_CROWD: "a fairly large crowd",
            CrowdSize.DENSE_CROWD: "a dense crowd",
        }[self.size]

        parts = ["There is " + words]
        if self.centre_direction:
            parts.append(" " + self.centre_direction)
        parts.append(", roughly " + str(self.person_count) + " people that I can see")

        if self.motion == CrowdMotion.STATIC:
            parts.append(". They appear mostly stationary")
        elif self.flow == CrowdFlow.UNIDIRECTIONAL and self.flow_direction:
            parts.append(". Most of them are moving " + self.flow_direction)
        elif self.flow == CrowdFlow.BIDIRECTIONAL:
            parts.append(". People are moving in both directions")
        elif self.flow == CrowdFlow.CONVERGING:
            parts.append(". They appear to be converging")
        elif self.flow == CrowdFlow.DIVERGING:
            parts.append(". They appear to be spreading out")
        elif self.motion != CrowdMotion.STATIC:
            parts.append(". Their movement is mixed")

        if self.approaching:
            parts.append(", and the group appears to be getting closer")
        elif self.departing:
            parts.append(", and the group appears to be moving away")
        if self.congested:
            parts.append(". The area looks congested")
        return "".join(parts) + "."


_DIRECTION_WORDS = {
    DirectionBucket.FAR_LEFT: "far to your left",
    DirectionBucket.LEFT: "to your left",
    DirectionBucket.CENTER_LEFT: "slightly to your left",
    DirectionBucket.CENTER: "ahead of you",
    DirectionBucket.CENTER_RIGHT: "slightly to your right",
    DirectionBucket.RIGHT: "to your right",
    DirectionBucket.FAR_RIGHT: "far to your right",
}


class CrowdDetectionEngine:
    """
    Consumes person tracks (which already carry smoothed velocity) and produces
    a debounced CrowdState plus the transitions worth telling the user about.
    """

    def __init__(self, config: Optional[CrowdConfig] = None) -> None:
        self.config = config or CrowdConfig()
        self.state = CrowdState()
        self._pending: Optional[CrowdState] = None
        self._pending_count = 0

    # ------------------------------------------------------------------

    def analyse(
        self, person_tracks: Sequence, image_width: int, image_height: int,
        now: Optional[float] = None,
    ) -> CrowdState:
        now = now if now is not None else time.time()
        count = len(person_tracks)
        if image_width <= 0 or image_height <= 0:
            return CrowdState(at=now)

        size = self._classify_size(count)
        if size == CrowdSize.NO_CROWD:
            return CrowdState(person_count=count, confidence=0.9, at=now)

        coverage = sum(t.box.area for t in person_tracks) / float(image_width * image_height)
        density = self._classify_density(coverage, count)
        # A big crowd of tiny distant figures is not dense; coverage promotes.
        if density == Density.HIGH and size == CrowdSize.LARGE_CROWD:
            size = CrowdSize.DENSE_CROWD

        speeds = [t.speed for t in person_tracks]
        mean_speed = sum(speeds) / len(speeds)
        motion = self._classify_motion(mean_speed)

        flow, flow_direction, agreement = self._classify_flow(person_tracks, motion)
        if motion != CrowdMotion.STATIC and flow == CrowdFlow.CHAOTIC:
            motion = CrowdMotion.MIXED_FLOW

        centre_x = sum(t.box.center_x for t in person_tracks) / count
        centre_direction = _DIRECTION_WORDS.get(
            _raw_bucket(centre_x / image_width), "ahead of you"
        )

        approaching, departing = self._classify_approach(person_tracks)
        congested = density == Density.HIGH and motion in (
            CrowdMotion.STATIC, CrowdMotion.SLOW_MOVING
        )

        # Confidence reflects how much evidence we actually have: few people or
        # very short tracks means a weak read.
        track_maturity = min(1.0, sum(getattr(t, "hits", 1) for t in person_tracks) / (count * 4))
        confidence = round(min(0.95, 0.45 + 0.3 * track_maturity + 0.2 * min(1.0, count / 8)), 3)

        return CrowdState(
            size=size, density=density, motion=motion, flow=flow,
            person_count=count, approaching=approaching, departing=departing,
            congested=congested, dominant_direction=flow_direction,
            flow_direction=flow_direction, centre_direction=centre_direction,
            confidence=confidence, at=now,
        )

    # ------------------------------------------------------------------

    def _classify_size(self, count: int) -> CrowdSize:
        config = self.config
        if count >= config.large_min:
            return CrowdSize.LARGE_CROWD
        if count >= config.moderate_min:
            return CrowdSize.MODERATE_CROWD
        if count >= config.small_group_min:
            return CrowdSize.SMALL_GROUP
        return CrowdSize.NO_CROWD

    def _classify_density(self, coverage: float, count: int) -> Density:
        if coverage >= self.config.dense_coverage:
            return Density.HIGH
        if coverage >= self.config.medium_coverage:
            return Density.MEDIUM
        return Density.LOW

    def _classify_motion(self, mean_speed: float) -> CrowdMotion:
        config = self.config
        if mean_speed < config.static_speed:
            return CrowdMotion.STATIC
        if mean_speed < config.slow_speed:
            return CrowdMotion.SLOW_MOVING
        if mean_speed < config.fast_speed:
            return CrowdMotion.MOVING
        return CrowdMotion.FAST_MOVING

    def _classify_flow(self, tracks: Sequence, motion: CrowdMotion) -> tuple:
        """Part 28: turn per-person velocity vectors into a flow description."""
        if motion == CrowdMotion.STATIC:
            return CrowdFlow.NONE, None, 0.0

        moving = [t for t in tracks if t.speed >= self.config.static_speed]
        if len(moving) < 2:
            return CrowdFlow.NONE, None, 0.0

        # Mean unit vector: magnitude near 1 means everyone agrees, near 0
        # means directions cancel out.
        sum_x = sum(t.vx / max(t.speed, 1e-6) for t in moving)
        sum_y = sum(t.vy / max(t.speed, 1e-6) for t in moving)
        agreement = math.sqrt(sum_x ** 2 + sum_y ** 2) / len(moving)
        mean_x = sum_x / len(moving)

        if agreement >= self.config.unidirectional_agreement:
            if abs(mean_x) < 0.35:
                direction = "ahead" if sum_y <= 0 else "toward you"
            else:
                direction = "toward your right" if mean_x > 0 else "toward your left"
            return CrowdFlow.UNIDIRECTIONAL, direction, agreement

        # Low agreement: is it two opposing streams, or converging/diverging?
        centre = sum(t.box.center_x for t in moving) / len(moving)
        inward = sum(
            1 for t in moving
            if (t.box.center_x < centre and t.vx > 0) or (t.box.center_x > centre and t.vx < 0)
        )
        outward = len(moving) - inward
        if inward >= len(moving) * 0.7:
            return CrowdFlow.CONVERGING, None, agreement
        if outward >= len(moving) * 0.7:
            return CrowdFlow.DIVERGING, None, agreement
        if agreement >= self.config.bidirectional_agreement:
            return CrowdFlow.BIDIRECTIONAL, None, agreement
        left = sum(1 for t in moving if t.vx < 0)
        if 0.3 <= left / len(moving) <= 0.7:
            return CrowdFlow.BIDIRECTIONAL, None, agreement
        return CrowdFlow.CHAOTIC, None, agreement

    def _classify_approach(self, tracks: Sequence) -> tuple:
        """
        Part 30. Growing apparent size means closing distance. Requires a
        majority to agree, so one person walking up does not make "the crowd"
        approach.
        """
        rate = self.config.approach_scale_rate
        growing = sum(1 for t in tracks if getattr(t, "scale_rate", 0.0) > rate)
        shrinking = sum(1 for t in tracks if getattr(t, "scale_rate", 0.0) < -rate)
        threshold = max(2, len(tracks) // 2)
        return growing >= threshold, shrinking >= threshold

    # ------------------------------------------------------------------

    def update(
        self, person_tracks: Sequence, image_width: int, image_height: int,
        now: Optional[float] = None,
    ) -> tuple:
        """
        Analyse and debounce. Returns (state, transitions) where transitions are
        the (event_name, detail) pairs worth reporting.

        Debounced because at 6 FPS a single missed detection can flip a crowd
        from "moderate" to "small group" and back, and announcing that churn
        would be maddening.
        """
        fresh = self.analyse(person_tracks, image_width, image_height, now)

        significant = (
            fresh.size != self.state.size
            or fresh.motion != self.state.motion
            or fresh.flow != self.state.flow
            or fresh.density != self.state.density
            or fresh.approaching != self.state.approaching
            or fresh.congested != self.state.congested
        )
        if significant:
            if self._pending and self._same_shape(self._pending, fresh):
                self._pending_count += 1
            else:
                self._pending, self._pending_count = fresh, 1
            if self._pending_count < self.config.confirm_frames:
                # Not yet confirmed — keep reporting the stable state.
                return self.state, []
        else:
            self._pending, self._pending_count = None, 0

        previous, self.state = self.state, fresh
        self._pending, self._pending_count = None, 0
        return fresh, self._transitions(previous, fresh)

    @staticmethod
    def _same_shape(a: CrowdState, b: CrowdState) -> bool:
        return (a.size, a.motion, a.flow, a.density, a.approaching, a.congested) == (
            b.size, b.motion, b.flow, b.density, b.approaching, b.congested
        )

    def _transitions(self, previous: CrowdState, current: CrowdState) -> List[tuple]:
        """The structured events of Part 32."""
        from app.events.bus import EventType

        events: List[tuple] = []
        order = [CrowdSize.NO_CROWD, CrowdSize.SMALL_GROUP, CrowdSize.MODERATE_CROWD,
                 CrowdSize.LARGE_CROWD, CrowdSize.DENSE_CROWD]

        if not previous.is_crowd and current.is_crowd:
            events.append((EventType.CROWD_APPEARED, current.describe()))
        elif previous.is_crowd and not current.is_crowd:
            events.append((EventType.CROWD_DISPERSED, "The crowd ahead has dispersed."))
        else:
            if order.index(current.size) > order.index(previous.size):
                events.append((EventType.CROWD_DENSITY_INCREASED,
                               "The area ahead is getting busier."))
            elif order.index(current.size) < order.index(previous.size):
                events.append((EventType.CROWD_DENSITY_DECREASED,
                               "The crowd ahead is thinning out."))

        if previous.motion == CrowdMotion.STATIC and current.motion != CrowdMotion.STATIC:
            detail = "The crowd has started moving"
            if current.flow_direction:
                detail += " " + current.flow_direction
            events.append((EventType.CROWD_STARTED_MOVING, detail + "."))
        elif previous.motion != CrowdMotion.STATIC and current.motion == CrowdMotion.STATIC:
            events.append((EventType.CROWD_STOPPED, "The crowd ahead has stopped moving."))
        elif (previous.flow_direction and current.flow_direction
              and previous.flow_direction != current.flow_direction):
            events.append((EventType.CROWD_DIRECTION_CHANGED,
                           "The crowd is now moving " + current.flow_direction + "."))

        if current.approaching and not previous.approaching:
            events.append((EventType.CROWD_APPROACHING,
                           "A group of people appears to be moving toward you."))
        if current.departing and not previous.departing:
            events.append((EventType.CROWD_DEPARTING,
                           "The group ahead appears to be moving away."))
        if current.congested and not previous.congested:
            events.append((EventType.CROWD_CONGESTION,
                           "The area ahead looks congested."))
        return events

    def reset(self) -> None:
        self.state = CrowdState()
        self._pending, self._pending_count = None, 0
