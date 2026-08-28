from app.perception.distance import (
    DistanceEstimator,
    DistanceRange,
    MIN_CONFIDENCE_TO_ASSERT,
    SmoothedDistanceEstimator,
)


def test_never_reports_fake_metric_precision() -> None:
    est = DistanceEstimator()
    result = est.estimate("person", box_height_px=300, image_height_px=480)
    # The label must be a qualitative phrase, never a bare decimal-meter number.
    assert "meters" in result.label or result.range == DistanceRange.UNKNOWN
    import re

    assert not re.search(r"\d+\.\d{2,}", result.label), "label must not contain fake precision digits"


def test_large_apparent_height_is_closer_than_small() -> None:
    est = DistanceEstimator()
    close = est.estimate("person", box_height_px=400, image_height_px=480)
    far = est.estimate("person", box_height_px=40, image_height_px=480)
    order = [DistanceRange.VERY_CLOSE, DistanceRange.CLOSE, DistanceRange.NEARBY, DistanceRange.FAR, DistanceRange.UNKNOWN]
    assert order.index(close.range) <= order.index(far.range)


def test_zero_or_negative_input_is_unknown_not_a_crash() -> None:
    est = DistanceEstimator()
    assert est.estimate("person", 0, 480).range == DistanceRange.UNKNOWN
    assert est.estimate("person", 100, 0).range == DistanceRange.UNKNOWN


def test_low_confidence_estimate_is_reported_as_unknown_not_asserted() -> None:
    est = DistanceEstimator()
    result = est.estimate("person", box_height_px=1, image_height_px=480)
    if result.range != DistanceRange.UNKNOWN:
        assert result.confidence >= MIN_CONFIDENCE_TO_ASSERT


def test_smoothed_estimator_rejects_a_single_outlier_reading() -> None:
    smoother = SmoothedDistanceEstimator()
    # Four consistent "close" readings...
    for _ in range(4):
        smoother.update("person", box_height_px=250, image_height_px=480)
    # ...then one wildly different single reading shouldn't flip the answer.
    result = smoother.update("person", box_height_px=10, image_height_px=480)
    assert result.range != DistanceRange.FAR and result.range != DistanceRange.UNKNOWN
