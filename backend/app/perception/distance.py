"""
Distance/depth estimation — Segment 6.

CRITICAL CONSTRAINT (see ARCHITECTURE.md / SAFETY POLICY): a monocular RGB
camera does not give real metric distance without calibrated hardware.
This module NEVER reports fake precision ("2.384729 meters"). It reports a
qualitative range with an explicit confidence, and refuses to assert a
range at all below a confidence floor — the caller is expected to speak
`DistanceEstimate.label` (or, when the range is UNKNOWN, say it can't
reliably estimate distance) rather than inventing a number.

Method (v1, this hardware): a bounding-box-height heuristic, normalized by
a per-class "typical real-world height" prior — an object's apparent pixel
height relative to the frame, divided by how tall that class of object
usually is, gives a monotonic (not metric) closeness signal. This is
deliberately not a depth model; it's the safest thing that can be built
today on CPU-only hardware without overclaiming precision. A real depth
model (MiDaS / Depth Anything) can replace `DistanceEstimator.estimate`
later without changing its signature or any caller.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, Optional


class DistanceRange(str, Enum):
    VERY_CLOSE = "very_close"  # almost within reach
    CLOSE = "close"  # roughly 1-2m
    NEARBY = "nearby"  # roughly 2-4m
    FAR = "far"  # past ~4m
    UNKNOWN = "unknown"  # confidence too low to assert anything


@dataclass
class DistanceEstimate:
    range: DistanceRange
    confidence: float
    method: str
    label: str


_RANGE_LABELS: Dict[DistanceRange, str] = {
    DistanceRange.VERY_CLOSE: "very close, almost within reach",
    DistanceRange.CLOSE: "approximately 1 to 2 meters away",
    DistanceRange.NEARBY: "approximately 2 to 4 meters away",
    DistanceRange.FAR: "farther away, past about 4 meters",
    DistanceRange.UNKNOWN: "at an uncertain distance",
}

# Rough typical real-world heights (meters), used only to normalize apparent
# bbox height across different object classes — NOT a metric depth model.
# A person and a cup filling the same fraction of frame are not the same
# distance away, hence per-class priors instead of one universal curve.
_TYPICAL_HEIGHT_M: Dict[str, float] = {
    "person": 1.6,
    "chair": 0.9,
    "dining table": 0.75,
    "backpack": 0.45,
    "handbag": 0.3,
    "suitcase": 0.6,
    "bottle": 0.25,
    "cell phone": 0.15,
    "laptop": 0.25,
    "book": 0.22,
    "cup": 0.1,
    "car": 1.5,
    "bus": 3.0,
    "truck": 2.5,
    "bicycle": 1.0,
    "motorcycle": 1.1,
}
_DEFAULT_TYPICAL_HEIGHT_M = 0.5

MIN_CONFIDENCE_TO_ASSERT = 0.35


class DistanceEstimator:
    """Single-frame estimate. See SmoothedDistanceEstimator for the per-track, temporally-smoothed version."""

    def estimate(self, class_name: str, box_height_px: float, image_height_px: float) -> DistanceEstimate:
        if box_height_px <= 0 or image_height_px <= 0:
            return self._unknown()

        typical_m = _TYPICAL_HEIGHT_M.get(class_name, _DEFAULT_TYPICAL_HEIGHT_M)
        apparent_fraction = min(box_height_px / image_height_px, 1.0)
        if apparent_fraction <= 0:
            return self._unknown()

        # Bigger apparent fraction per unit of typical real-world size ==
        # closer. Unitless, monotonic, deliberately not claiming metric
        # accuracy — thresholds below are qualitative, not measured.
        closeness = apparent_fraction / typical_m

        if closeness > 1.1:
            range_, confidence = DistanceRange.VERY_CLOSE, 0.55
        elif closeness > 0.55:
            range_, confidence = DistanceRange.CLOSE, 0.5
        elif closeness > 0.22:
            range_, confidence = DistanceRange.NEARBY, 0.45
        else:
            range_, confidence = DistanceRange.FAR, 0.4

        if confidence < MIN_CONFIDENCE_TO_ASSERT:
            return self._unknown(confidence)

        return DistanceEstimate(range_, confidence, "bbox_height_heuristic_v1", _RANGE_LABELS[range_])

    def _unknown(self, confidence: float = 0.0) -> DistanceEstimate:
        return DistanceEstimate(DistanceRange.UNKNOWN, confidence, "bbox_height_heuristic_v1", _RANGE_LABELS[DistanceRange.UNKNOWN])


class SmoothedDistanceEstimator:
    """
    Per-track wrapper adding temporal smoothing + outlier rejection: one
    noisy single-frame reading doesn't get to redefine the answer. Keep one
    instance per tracked object, matching DirectionEstimator's per-track
    hysteresis pattern.
    """

    def __init__(self, estimator: Optional[DistanceEstimator] = None, history_size: int = 5) -> None:
        self._estimator = estimator or DistanceEstimator()
        self._history: Deque[DistanceEstimate] = deque(maxlen=history_size)

    def update(self, class_name: str, box_height_px: float, image_height_px: float) -> DistanceEstimate:
        raw = self._estimator.estimate(class_name, box_height_px, image_height_px)
        self._history.append(raw)

        ranges = [e.range for e in self._history]
        smoothed_range, _count = Counter(ranges).most_common(1)[0]
        agreeing = [e.confidence for e in self._history if e.range == smoothed_range]
        smoothed_confidence = round(sum(agreeing) / len(agreeing), 3)

        return DistanceEstimate(
            range=smoothed_range,
            confidence=smoothed_confidence,
            method=raw.method + "+temporal_smoothing",
            label=_RANGE_LABELS[smoothed_range],
        )

    def reset(self) -> None:
        self._history.clear()
