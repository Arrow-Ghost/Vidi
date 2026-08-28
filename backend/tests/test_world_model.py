"""
Segment 4 acceptance test. Deliberately an integration test across
Segments 3 (tracker), 5 (direction), and 6 (distance) using synthetic
detections — this is the seam the agent will actually call, so it needs to
be tested as the seam, not just as isolated units.
"""

from app.perception.object_detector import BoundingBox, Detection
from app.perception.tracker import ObjectTracker
from app.perception.world_model import WorldModel

IMG_W, IMG_H = 640, 480


def make_detection(class_name: str, x1, y1, x2, y2, conf=0.9) -> Detection:
    return Detection(
        class_name=class_name,
        confidence=conf,
        box=BoundingBox(x1, y1, x2, y2),
        timestamp=0.0,
        image_width=IMG_W,
        image_height=IMG_H,
    )


def test_apply_frame_populates_entities_with_direction_and_distance() -> None:
    world = WorldModel()
    tracker = ObjectTracker()
    detections = [make_detection("backpack", 500, 200, 560, 300)]  # right side of frame

    world.apply_frame(tracker, detections, IMG_W, IMG_H)

    assert len(world.entities) == 1
    entity = next(iter(world.entities.values()))
    assert entity.type == "backpack"
    assert entity.direction.bucket.value in ("right", "far_right", "center_right")
    assert entity.distance.label  # always a phrase, never absent


def test_object_leaving_frame_is_removed_from_world_model() -> None:
    world = WorldModel()
    tracker = ObjectTracker()
    world.apply_frame(tracker, [make_detection("bottle", 0, 0, 20, 20)], IMG_W, IMG_H)
    assert len(world.entities) == 1

    # Enough consecutive empty frames to cross max_misses_before_lost (default 5).
    for _ in range(6):
        world.apply_frame(tracker, [], IMG_W, IMG_H)

    assert len(world.entities) == 0


def test_summarize_prioritizes_requested_class_then_person_then_other() -> None:
    world = WorldModel()
    tracker = ObjectTracker()
    detections = [
        make_detection("chair", 0, 0, 40, 40),
        make_detection("person", 100, 0, 140, 200),
        make_detection("backpack", 300, 0, 340, 100),
        make_detection("cup", 500, 0, 520, 40),
    ]
    world.apply_frame(tracker, detections, IMG_W, IMG_H)

    summary = world.summarize(requested_class="backpack", max_items=4)
    types_in_order = [e.type for e in summary]
    assert types_in_order[0] == "backpack"  # requested object always first
    assert "person" in types_in_order[:2]  # people rank above generic objects


def test_summarize_caps_at_max_items_even_with_more_tracked() -> None:
    world = WorldModel()
    tracker = ObjectTracker()
    detections = [make_detection("cup", i * 10, 0, i * 10 + 8, 8) for i in range(10)]
    world.apply_frame(tracker, detections, IMG_W, IMG_H)

    assert len(world.entities) == 10  # all tracked...
    assert len(world.summarize(max_items=3)) == 3  # ...but summary is capped


def test_find_returns_only_matching_class() -> None:
    world = WorldModel()
    tracker = ObjectTracker()
    detections = [
        make_detection("backpack", 0, 0, 40, 40),
        make_detection("cup", 100, 0, 120, 20),
    ]
    world.apply_frame(tracker, detections, IMG_W, IMG_H)

    backpacks = world.find("backpack")
    assert len(backpacks) == 1
    assert backpacks[0].type == "backpack"
