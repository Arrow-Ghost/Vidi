"""
Direction engine — Segment 5.

Maps a detection's bounding-box center to a relative spatial bucket (7-way,
per the brief's preferred granularity over plain left/center/right) from
bbox center position and image width. Per-track hysteresis prevents the
"LEFT, CENTER, LEFT, CENTER" flicker the brief explicitly calls out for an
object sitting near a bucket boundary — use one DirectionEstimator instance
per tracked object (keyed by track ID) so each object's flicker filtering
is independent of every other object's.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DirectionBucket(str, Enum):
    FAR_LEFT = "far_left"
    LEFT = "left"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    RIGHT = "right"
    FAR_RIGHT = "far_right"


_BUCKETS = [
    DirectionBucket.FAR_LEFT,
    DirectionBucket.LEFT,
    DirectionBucket.CENTER_LEFT,
    DirectionBucket.CENTER,
    DirectionBucket.CENTER_RIGHT,
    DirectionBucket.RIGHT,
    DirectionBucket.FAR_RIGHT,
]
# 8 boundary fractions (0.0 .. 1.0) carving the frame into the 7 buckets above.
_BOUNDARIES = [i / 7 for i in range(8)]

# A value must land at least this far (as a fraction of image width) past a
# boundary before the reported bucket is allowed to switch. Prevents a
# center_x oscillating by a few pixels right at a boundary from flipping
# the reported direction every frame.
DEFAULT_HYSTERESIS_MARGIN = 0.03


@dataclass
class DirectionEstimate:
    bucket: DirectionBucket
    confidence: float


def _raw_bucket(fraction: float) -> DirectionBucket:
    fraction = min(max(fraction, 0.0), 1.0)
    for i in range(7):
        if fraction <= _BOUNDARIES[i + 1]:
            return _BUCKETS[i]
    return _BUCKETS[-1]


def _distance_to_nearest_boundary(fraction: float) -> float:
    return min(abs(fraction - b) for b in _BOUNDARIES)


class DirectionEstimator:
    """One instance per tracked object — this is what makes hysteresis per-object."""

    def __init__(self, hysteresis_margin: float = DEFAULT_HYSTERESIS_MARGIN) -> None:
        self._current: Optional[DirectionBucket] = None
        self._margin = hysteresis_margin

    def update(self, center_x: float, image_width: float) -> DirectionEstimate:
        if image_width <= 0:
            raise ValueError("image_width must be > 0")
        fraction = center_x / image_width
        raw = _raw_bucket(fraction)
        boundary_gap = _distance_to_nearest_boundary(fraction)

        if self._current is None:
            self._current = raw
        elif raw != self._current and boundary_gap >= self._margin:
            # Clearly past the boundary, not just noise — allow the switch.
            self._current = raw
        # else: keep the previous bucket; this reading is boundary noise.

        # Confidence reflects how far the reading sits from the nearest
        # bucket boundary — near a boundary, a slightly different bbox
        # estimate would have landed in the neighboring bucket instead.
        confidence = round(min(1.0, 0.5 + boundary_gap * 4), 3)
        return DirectionEstimate(bucket=self._current, confidence=confidence)

    def reset(self) -> None:
        self._current = None
