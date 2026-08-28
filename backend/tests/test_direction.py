from app.perception.direction import DirectionBucket, DirectionEstimator


def test_center_of_frame_is_center_bucket() -> None:
    est = DirectionEstimator()
    result = est.update(center_x=320, image_width=640)
    assert result.bucket == DirectionBucket.CENTER


def test_far_left_and_far_right_edges() -> None:
    assert DirectionEstimator().update(0, 640).bucket == DirectionBucket.FAR_LEFT
    assert DirectionEstimator().update(640, 640).bucket == DirectionBucket.FAR_RIGHT


def test_hysteresis_prevents_flicker_near_boundary() -> None:
    est = DirectionEstimator(hysteresis_margin=0.03)
    # Boundary between CENTER_LEFT and CENTER sits at 3/7 * 640 ≈ 274.3px.
    boundary_x = (3 / 7) * 640

    first = est.update(boundary_x - 1, 640)  # just left of boundary
    # Oscillate right at the boundary a few times — should NOT flip every call.
    for _ in range(5):
        wobble = est.update(boundary_x + 1, 640)
        assert wobble.bucket == first.bucket, "small boundary noise must not flip the reported bucket"

    # A clear, sustained move well past the margin DOES switch it.
    moved = est.update(boundary_x + 40, 640)
    assert moved.bucket != first.bucket


def test_confidence_is_lower_near_a_boundary_than_at_bucket_center() -> None:
    est_near = DirectionEstimator()
    boundary_x = (3 / 7) * 640
    near = est_near.update(boundary_x + 0.5, 640)

    est_mid = DirectionEstimator()
    mid = est_mid.update(320, 640)  # dead center of the CENTER bucket

    assert near.confidence < mid.confidence


def test_rejects_non_positive_image_width() -> None:
    import pytest

    with pytest.raises(ValueError):
        DirectionEstimator().update(10, 0)
