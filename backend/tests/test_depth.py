"""
Depth tests — Segment 6.

The headline test asserts against real-world geometry a human can verify by
looking at the photograph: in bus.jpg the people stand IN FRONT OF the bus,
so the depth model must rank them nearer. That is a far stronger check than
asserting the model merely returns numbers.

The rest cover the honesty properties: conflicting evidence must produce
UNKNOWN, stale depth must be ignored, and a missing model must degrade to the
size prior rather than crash or fabricate.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.perception.depth import (
    CONF_ADJACENT,
    CONF_AGREE,
    DepthEstimator,
    DepthMap,
    FusedDistanceEstimator,
    depth_percentile,
    fuse,
    sample_box_depth,
)
from app.perception.distance import DistanceEstimate, DistanceRange
from app.perception.object_detector import BoundingBox, ObjectDetector

FIXTURES = Path(__file__).parent / "fixtures"
DEPTH_MODEL = (
    Path(__file__).resolve().parents[2] / "models" / "weights" / "depth_anything_v2_small.onnx"
)
needs_model = pytest.mark.skipif(not DEPTH_MODEL.exists(), reason="depth model not downloaded")


def _frame(name: str = "bus.jpg") -> np.ndarray:
    image = cv2.imread(str(FIXTURES / name))
    assert image is not None
    return cv2.resize(image, (640, 480))


def _fake_map(values: np.ndarray, age_s: float = 0.0) -> DepthMap:
    return DepthMap(values=values.astype(np.float32),
                    created_at=time.time() - age_s, inference_ms=1.0)


# ---------------- the model itself ----------------

@needs_model
def test_depth_output_polarity_matches_real_geometry() -> None:
    """
    Larger value must mean NEARER. Verified against known scene geometry: the
    ground at the bottom of the frame is nearer than the sky at the top.
    Getting this backwards would invert every distance the system reports.
    """
    depth_map = DepthEstimator().infer(_frame())
    assert depth_map is not None
    values = depth_map.values
    height, width = values.shape

    bottom = np.median(values[int(0.88 * height):, int(0.35 * width):int(0.65 * width)])
    top = np.median(values[: int(0.12 * height), int(0.35 * width):int(0.65 * width)])
    assert bottom > top, "depth polarity is inverted — larger must mean nearer"


@needs_model
def test_people_are_ranked_nearer_than_the_bus_behind_them() -> None:
    """Ground truth anyone can check by opening bus.jpg."""
    frame = _frame()
    detections = ObjectDetector().detect(frame)
    depth_map = DepthEstimator().infer(frame)
    assert depth_map is not None

    def percentile_for(detection) -> float:
        raw = sample_box_depth(depth_map, detection.box, 640, 480)
        assert raw is not None
        return depth_percentile(depth_map, raw)

    people = [percentile_for(d) for d in detections if d.class_name == "person"]
    buses = [percentile_for(d) for d in detections if d.class_name == "bus"]
    assert people and buses, "fixture should contain both people and a bus"

    # The nearest person must read as nearer than the bus they stand in front of.
    assert max(people) > buses[0]


@needs_model
def test_depth_distinguishes_objects_the_size_prior_cannot() -> None:
    """
    The whole point of adding depth: three people of similar apparent size
    collapse to one bucket under the size prior alone. Depth must spread them.
    """
    frame = _frame()
    detections = ObjectDetector().detect(frame)
    depth_map = DepthEstimator().infer(frame)
    people = [d for d in detections if d.class_name == "person"]
    assert len(people) >= 2

    percentiles = [
        depth_percentile(depth_map, sample_box_depth(depth_map, d.box, 640, 480))
        for d in people
    ]
    assert max(percentiles) - min(percentiles) > 0.2, (
        "depth gave near-identical readings for people at visibly different distances"
    )


def test_missing_model_degrades_instead_of_crashing() -> None:
    estimator = DepthEstimator(model_path=Path("does-not-exist.onnx"))
    assert estimator.available is False
    assert estimator.infer(np.zeros((480, 640, 3), dtype=np.uint8)) is None


# ---------------- sampling ----------------

def test_box_sampling_uses_the_centre_and_ignores_background_edges() -> None:
    """
    A tall thin object's box is mostly background. Sampling the whole box
    would report the background's depth; sampling the centre reports the
    object's.
    """
    values = np.zeros((224, 224), dtype=np.float32)  # far background
    values[80:144, 96:128] = 9.0                     # near object in the middle
    depth_map = _fake_map(values)

    box = BoundingBox(x1=260, y1=160, x2=380, y2=320)  # centred on the object
    reading = sample_box_depth(depth_map, box, 640, 480)
    assert reading == pytest.approx(9.0), "centre sampling picked up background"


def test_percentile_is_self_normalizing() -> None:
    """Depth's scale drifts between frames; percentile rank must not."""
    small = _fake_map(np.linspace(0, 1, 224 * 224).reshape(224, 224))
    large = _fake_map(np.linspace(0, 1000, 224 * 224).reshape(224, 224))
    assert depth_percentile(small, 0.5) == pytest.approx(depth_percentile(large, 500), abs=0.02)


# ---------------- fusion honesty ----------------

def _prior(range_: DistanceRange, confidence: float = 0.45) -> DistanceEstimate:
    return DistanceEstimate(range=range_, confidence=confidence,
                            method="bbox_height_heuristic_v1", label="x")


def test_agreement_raises_confidence() -> None:
    result = fuse(DistanceRange.CLOSE, _prior(DistanceRange.CLOSE))
    assert result.range == DistanceRange.CLOSE
    assert result.confidence == CONF_AGREE
    assert "agree" in result.method


def test_adjacent_disagreement_trusts_depth_but_lowers_confidence() -> None:
    result = fuse(DistanceRange.CLOSE, _prior(DistanceRange.NEARBY))
    assert result.range == DistanceRange.CLOSE
    assert result.confidence == CONF_ADJACENT


def test_gross_disagreement_reports_unknown_rather_than_picking_a_side() -> None:
    """
    THE critical honesty property. Two independent estimates that contradict
    each other are evidence neither should be spoken aloud.
    """
    result = fuse(DistanceRange.VERY_CLOSE, _prior(DistanceRange.FAR))
    assert result.range == DistanceRange.UNKNOWN
    assert "conflict" in result.method


def test_depth_alone_is_used_when_the_size_prior_is_unknown() -> None:
    result = fuse(DistanceRange.NEARBY, _prior(DistanceRange.UNKNOWN, 0.0))
    assert result.range == DistanceRange.NEARBY
    assert result.confidence == CONF_ADJACENT


# ---------------- estimator lifecycle ----------------

def test_estimator_falls_back_to_size_prior_without_a_depth_map() -> None:
    estimator = FusedDistanceEstimator()
    box = BoundingBox(x1=100, y1=100, x2=200, y2=400)
    result = estimator.update("person", box, 640, 480, depth_map=None)
    assert "bbox_height_heuristic" in result.method
    assert result.range != DistanceRange.UNKNOWN


def test_stale_depth_map_is_ignored() -> None:
    """
    A depth map from several seconds ago describes a scene the user has since
    walked away from. It must not be fused as if current.
    """
    estimator = FusedDistanceEstimator()
    box = BoundingBox(x1=100, y1=100, x2=200, y2=400)
    stale = _fake_map(np.full((224, 224), 5.0), age_s=30.0)
    result = estimator.update("person", box, 640, 480, stale, max_depth_age_s=1.5)
    assert "bbox_height_heuristic" in result.method, "stale depth map was used"


def test_temporal_smoothing_resists_a_single_noisy_frame() -> None:
    estimator = FusedDistanceEstimator()
    box = BoundingBox(x1=100, y1=100, x2=200, y2=400)
    steady = _fake_map(np.full((224, 224), 1.0))
    for _ in range(4):
        estimator.update("person", box, 640, 480, steady)
    stable = estimator.update("person", box, 640, 480, steady)

    spike = _fake_map(np.concatenate([
        np.full((112, 224), 0.0), np.full((112, 224), 99.0)
    ]).astype(np.float32))
    after = estimator.update("person", box, 640, 480, spike)
    assert after.range == stable.range, "one noisy frame changed the reported range"
