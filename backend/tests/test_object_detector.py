"""
Segment 3 acceptance test — real YOLOv8n inference, not a mock.

Runs actual detection against two real photographs (bundled with the
ultralytics package for exactly this kind of smoke test) and asserts on
real output: at least one person is found in both, a bus is found in the
bus photo, every confidence is a real number in (0, 1], and inference
latency is recorded so it can be reported honestly rather than guessed at.
"""

from __future__ import annotations

from pathlib import Path

from app.perception.object_detector import ObjectDetector

FIXTURES = Path(__file__).parent / "fixtures"


def test_detects_people_in_zidane_photo() -> None:
    detector = ObjectDetector()
    detections = detector.detect(str(FIXTURES / "zidane.jpg"))

    assert detections, "expected at least one detection in zidane.jpg"
    people = [d for d in detections if d.class_name == "person"]
    assert people, "expected at least one person detected"
    for det in detections:
        assert 0.0 < det.confidence <= 1.0
        assert det.box.width > 0 and det.box.height > 0

    assert detector.last_inference_seconds is not None
    print(f"\n[Segment 3] zidane.jpg inference: {detector.last_inference_seconds:.3f}s, "
          f"{len(detections)} detections")


def test_detects_person_and_bus_in_bus_photo() -> None:
    detector = ObjectDetector()
    detections = detector.detect(str(FIXTURES / "bus.jpg"))

    classes_found = {d.class_name for d in detections}
    assert "person" in classes_found
    assert "bus" in classes_found

    print(f"\n[Segment 3] bus.jpg inference: {detector.last_inference_seconds:.3f}s, "
          f"classes found: {sorted(classes_found)}")


def test_confidence_threshold_is_respected() -> None:
    detector = ObjectDetector(confidence_threshold=0.9)
    detections = detector.detect(str(FIXTURES / "zidane.jpg"))
    for det in detections:
        assert det.confidence >= 0.9


def test_irrelevant_classes_are_filtered_out() -> None:
    # bus.jpg contains no relevant-list class besides person/bus, but this
    # guards the filtering contract even if the model finds e.g. "tie".
    from app.perception.object_detector import RELEVANT_CLASSES

    detector = ObjectDetector()
    detections = detector.detect(str(FIXTURES / "bus.jpg"))
    for det in detections:
        assert det.class_name in RELEVANT_CLASSES
