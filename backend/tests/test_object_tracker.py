"""
Segment 3 acceptance test — ObjectTracker, using synthetic detections (no
model/image needed; this is pure tracking-logic verification, deliberately
isolated from ObjectDetector so a tracking bug and a detection bug can never
be confused with each other).
"""

from __future__ import annotations

from app.perception.object_detector import BoundingBox, Detection
from app.perception.tracker import ObjectTracker, TrackEvent


def make_detection(class_name: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Detection:
    return Detection(
        class_name=class_name,
        confidence=conf,
        box=BoundingBox(x1, y1, x2, y2),
        timestamp=0.0,
        image_width=640,
        image_height=480,
    )


def test_new_object_emits_entered_with_stable_id() -> None:
    tracker = ObjectTracker()
    events = tracker.update([make_detection("backpack", 10, 10, 50, 50)])
    assert len(events) == 1
    assert events[0].event == TrackEvent.ENTERED
    track_id = events[0].track_id

    # Same object, barely moved -> same track ID, UPDATED not a new ENTERED.
    events2 = tracker.update([make_detection("backpack", 11, 11, 51, 51)])
    assert len(events2) == 1
    assert events2[0].track_id == track_id
    assert events2[0].event == TrackEvent.UPDATED


def test_large_movement_emits_moved_same_id() -> None:
    tracker = ObjectTracker()
    events = tracker.update([make_detection("person", 0, 0, 40, 40)])
    track_id = events[0].track_id

    # dx=20: center moves exactly 20px (>= move_threshold_px) while IoU stays
    # at 0.333 (>= iou_match_threshold 0.3) — chosen deliberately so this
    # exercises the MOVED path rather than accidentally creating a new track.
    events2 = tracker.update([make_detection("person", 20, 0, 60, 40)])
    assert events2[0].track_id == track_id
    assert events2[0].event == TrackEvent.MOVED


def test_disappearance_then_left_after_threshold() -> None:
    tracker = ObjectTracker()
    events = tracker.update([make_detection("bottle", 0, 0, 20, 20)])
    track_id = events[0].track_id

    seen_lost = False
    for _ in range(10):
        events = tracker.update([])  # nothing detected this frame
        for e in events:
            if e.track_id == track_id and e.event == TrackEvent.LOST:
                seen_lost = True
            if e.track_id == track_id and e.event == TrackEvent.LEFT:
                assert seen_lost, "LEFT should be preceded by at least one LOST"
                assert track_id not in tracker.tracks
                return
    raise AssertionError("expected a LEFT event within 10 empty frames")


def test_brief_occlusion_does_not_immediately_declare_lost() -> None:
    tracker = ObjectTracker()
    events = tracker.update([make_detection("cup", 0, 0, 20, 20)])
    track_id = events[0].track_id

    # One missed frame (brief occlusion) — should NOT emit LOST yet.
    events = tracker.update([])
    assert all(e.track_id != track_id for e in events)
    assert track_id in tracker.tracks


def test_different_classes_never_match_even_with_full_overlap() -> None:
    tracker = ObjectTracker()
    e1 = tracker.update([make_detection("chair", 0, 0, 50, 50)])
    e2 = tracker.update([make_detection("backpack", 0, 0, 50, 50)])
    assert e1[0].track_id != e2[0].track_id
    assert e2[0].event == TrackEvent.ENTERED


def test_duplicate_overlapping_same_class_boxes_are_suppressed() -> None:
    tracker = ObjectTracker()
    events = tracker.update(
        [
            make_detection("person", 0, 0, 100, 100, conf=0.6),
            make_detection("person", 2, 2, 98, 98, conf=0.95),  # near-identical box, higher conf
        ]
    )
    assert len(events) == 1
    assert events[0].confidence == 0.95
