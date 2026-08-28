"""
Obstacle proximity and door/exit detection.

The negative assertions carry the weight here. It is easy to make an assistant
shout about every chair it sees; the hard parts are staying quiet about things
that are not in the way, never claiming a path is clear, and never guessing
which door is the exit.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app.agent.tools import ToolRegistry
from app.events.bus import EventType
from app.perception.direction import DirectionBucket, DirectionEstimate
from app.perception.distance import DistanceEstimate, DistanceRange
from app.perception.doors import (
    CONFIDENT_DOOR,
    Door,
    DoorState,
    DoorTracker,
    _direction_from_box,
)
from app.perception.obstacles import (
    MIN_DETECTION_CONFIDENCE,
    MIN_DISTANCE_CONFIDENCE,
    REPEAT_COOLDOWN_S,
    ObstacleMonitor,
)
from app.perception.world_model import WorldModel

FIXTURES = Path(__file__).parent / "fixtures"


def run(coro):
    return asyncio.run(coro)


@dataclass
class FakeEntity:
    """Stands in for a WorldModel entity."""
    id: str
    type: str
    direction: object
    distance: object
    confidence: float = 0.9
    is_person: bool = False


def entity(entity_id="chair_1", kind="chair",
           bucket=DirectionBucket.CENTER,
           distance_range=DistanceRange.VERY_CLOSE,
           detection_conf=0.9, distance_conf=0.7) -> FakeEntity:
    return FakeEntity(
        id=entity_id,
        type=kind,
        direction=DirectionEstimate(bucket=bucket, confidence=0.9),
        distance=DistanceEstimate(
            range=distance_range, confidence=distance_conf,
            method="test", label="approximately 1 to 2 meters away",
        ),
        confidence=detection_conf,
        is_person=(kind == "person"),
    )


# ---------------- what counts as an obstacle ----------------

def test_a_close_chair_directly_ahead_is_an_urgent_alert():
    monitor = ObstacleMonitor()
    found = monitor.scan([entity()])
    assert len(found) == 1
    assert found[0].urgent is True
    assert "directly ahead" in found[0].describe()


def test_a_person_ahead_is_reported_as_someone_not_by_class_name():
    monitor = ObstacleMonitor()
    found = monitor.scan([entity("person_1", "person")])
    assert found[0].spoken_name == "someone"
    assert "Someone directly ahead" in found[0].describe()


def test_something_off_to_the_side_is_not_in_the_way():
    """Close, but not on a collision course — must not interrupt."""
    monitor = ObstacleMonitor()
    assert monitor.scan([entity(bucket=DirectionBucket.FAR_LEFT)]) == []
    assert monitor.scan([entity(bucket=DirectionBucket.RIGHT)]) == []


def test_distant_objects_are_not_obstacles():
    monitor = ObstacleMonitor()
    assert monitor.scan([entity(distance_range=DistanceRange.NEARBY)]) == []
    assert monitor.scan([entity(distance_range=DistanceRange.FAR)]) == []


def test_a_weak_detection_never_raises_a_hazard():
    monitor = ObstacleMonitor()
    assert monitor.scan([entity(detection_conf=MIN_DETECTION_CONFIDENCE - 0.1)]) == []


def test_an_unreliable_distance_never_raises_a_hazard():
    """
    We may think it is close, but if depth is not confident that is a guess,
    and guessing produces false alarms that train the user to ignore us.
    """
    monitor = ObstacleMonitor()
    assert monitor.scan([entity(distance_conf=MIN_DISTANCE_CONFIDENCE - 0.1)]) == []


def test_unknown_distance_is_never_an_obstacle():
    monitor = ObstacleMonitor()
    assert monitor.scan([entity(distance_range=DistanceRange.UNKNOWN)]) == []


def test_harmless_objects_are_not_hazards():
    monitor = ObstacleMonitor()
    assert monitor.scan([entity("banana_1", "banana")]) == []
    assert monitor.scan([entity("cup_1", "cup")]) == []


def test_closest_is_reported_first():
    monitor = ObstacleMonitor()
    found = monitor.scan([
        entity("chair_1", "chair", distance_range=DistanceRange.CLOSE),
        entity("person_1", "person", distance_range=DistanceRange.VERY_CLOSE),
    ])
    assert found[0].type == "person"


# ---------------- not nagging ----------------

def test_the_same_chair_is_not_announced_repeatedly():
    clock = {"t": 1000.0}
    monitor = ObstacleMonitor(clock=lambda: clock["t"])
    entities = [entity()]

    assert len(monitor.poll(entities)) == 1
    for _ in range(5):
        assert monitor.poll(entities) == []
    assert monitor.suppressed >= 5


def test_it_speaks_again_once_the_cooldown_passes():
    clock = {"t": 1000.0}
    monitor = ObstacleMonitor(clock=lambda: clock["t"])
    entities = [entity()]
    monitor.poll(entities)
    clock["t"] += REPEAT_COOLDOWN_S + 1
    assert len(monitor.poll(entities)) == 1


def test_getting_closer_re_alerts_immediately():
    """Approaching something is new information and must not wait for a cooldown."""
    clock = {"t": 1000.0}
    monitor = ObstacleMonitor(clock=lambda: clock["t"])

    monitor.poll([entity(distance_range=DistanceRange.CLOSE)])
    clock["t"] += 1
    escalated = monitor.poll([entity(distance_range=DistanceRange.VERY_CLOSE)])
    assert len(escalated) == 1
    assert escalated[0].urgent is True


def test_an_object_that_leaves_and_returns_alerts_again():
    clock = {"t": 1000.0}
    monitor = ObstacleMonitor(clock=lambda: clock["t"])
    monitor.poll([entity()])
    monitor.poll([])                       # it went away
    assert len(monitor.poll([entity()])) == 1


# ---------------- the tool must never promise a clear path ----------------

class _Pipeline:
    def __init__(self, entities=(), doors=()):
        self.world = WorldModel()
        for e in entities:
            self.world.entities[e.id] = e
        self.obstacle_monitor = ObstacleMonitor()
        self.door_tracker = DoorTracker()
        for d in doors:
            self.door_tracker.doors[d.key] = d
        self.ocr_available = False
        self.doors_available = False

    def current_frame(self):
        return None


def test_check_path_never_says_the_way_is_clear():
    registry = ToolRegistry(WorldModel(), pipeline=_Pipeline())
    result = run(registry.execute("check_path", {}))
    assert result["obstacle_count"] == 0
    spoken = (result["message"] + " " + result["note"]).lower()
    assert "clear" not in result["message"].lower()
    assert "does not mean the way ahead is clear" in spoken
    assert "never tell the user a path is clear" in spoken


def test_check_path_reports_the_closest_obstacle():
    pipeline = _Pipeline([entity("chair_1", "chair")])
    registry = ToolRegistry(WorldModel(), pipeline=pipeline)
    result = run(registry.execute("check_path", {}))
    assert result["obstacle_count"] == 1
    assert result["closest"]["type"] == "chair"
    assert "clear" in result["note"].lower()      # the warning is still attached


# ---------------- doors ----------------

def door(centre=0.5, state=DoorState.CLOSED, is_exit=False, confidence=0.95,
         kind="door", seen=None) -> Door:
    import time as _time

    from app.perception.direction import _raw_bucket

    stamp = _time.time() if seen is None else seen
    return Door(
        kind=kind, state=state, is_exit=is_exit,
        evidence="test", confidence=confidence,
        direction=_raw_bucket(centre), centre_x=centre,
        height_fraction=0.4, first_seen=stamp, last_seen=stamp,
    )


def test_box_maps_to_the_correct_side():
    left, cx, _ = _direction_from_box([200, 50, 800, 200])
    assert "left" in left.value and cx < 0.4
    right, cx, _ = _direction_from_box([200, 800, 800, 950])
    assert "right" in right.value and cx > 0.6


def test_a_new_door_is_reported_once():
    tracker = DoorTracker()
    events = tracker.update([door(0.5)], now=1000.0)
    assert len(events) == 1
    # Seeing it again in the same state is not news.
    assert tracker.update([door(0.5)], now=1001.0) == []


def test_a_door_opening_is_reported():
    tracker = DoorTracker()
    tracker.update([door(0.5, DoorState.CLOSED)], now=1000.0)
    events = tracker.update([door(0.5, DoorState.OPEN)], now=1005.0)
    assert len(events) == 1
    assert "has opened" in events[0][2]


def test_a_door_closing_is_reported():
    tracker = DoorTracker()
    tracker.update([door(0.5, DoorState.OPEN)], now=1000.0)
    events = tracker.update([door(0.5, DoorState.CLOSED)], now=1005.0)
    assert "has closed" in events[0][2]


def test_an_unknown_state_does_not_fake_a_transition():
    """A read that could not determine the state must not become 'it opened'."""
    tracker = DoorTracker()
    tracker.update([door(0.5, DoorState.OPEN)], now=1000.0)
    events = tracker.update([door(0.5, DoorState.UNKNOWN)], now=1005.0)
    assert events == []


def test_doors_at_different_positions_are_different_doors():
    tracker = DoorTracker()
    events = tracker.update([door(0.15), door(0.85)], now=1000.0)
    assert len(events) == 2
    assert len(tracker.current(now=1000.0)) == 2


def test_a_low_confidence_door_is_hedged():
    weak = door(0.5, confidence=CONFIDENT_DOOR - 0.2)
    assert weak.is_confident is False
    assert weak.describe().startswith("There may be")
    strong = door(0.5, confidence=0.95)
    assert strong.describe().startswith("There's")


def test_exits_are_listed_before_ordinary_doors():
    tracker = DoorTracker()
    tracker.update([door(0.2), door(0.8, is_exit=True)], now=1000.0)
    assert tracker.current(now=1000.0)[0].is_exit is True
    assert len(tracker.exits(now=1000.0)) == 1


def test_find_exit_refuses_to_guess_when_nothing_is_marked():
    pipeline = _Pipeline(doors=[door(0.2), door(0.8)])
    registry = ToolRegistry(WorldModel(), pipeline=pipeline)
    result = run(registry.execute("find_exit", {}))

    assert result["found"] is False
    assert result["doors_seen"] == 2
    assert "do not guess" in result["note"].lower()
    assert "haven't confirmed an exit" in result["message"].lower()


def test_find_exit_reports_a_marked_exit_with_its_evidence():
    marked = door(0.5, DoorState.OPEN, is_exit=True)
    marked.evidence = "green EXIT sign above the doorway"
    registry = ToolRegistry(WorldModel(), pipeline=_Pipeline(doors=[marked]))
    result = run(registry.execute("find_exit", {}))

    assert result["found"] is True
    assert "EXIT sign" in result["evidence"]
    assert "do not promise" in result["note"].lower()


def test_describe_doors_says_out_of_view_not_absent():
    registry = ToolRegistry(WorldModel(), pipeline=_Pipeline())
    result = run(registry.execute("describe_doors", {}))
    assert result["door_count"] == 0
    assert "out of view" in result["message"].lower()


# ---------------- real detection against the fixture ----------------

@pytest.mark.skipif(
    not (FIXTURES / "corridor_doors.jpg").exists(),
    reason="corridor fixture missing",
)
def test_yolo_genuinely_cannot_see_doors():
    """
    Documents WHY doors go through Gemini: the local detector returns nothing
    at all on a corridor full of them.
    """
    import cv2

    from app.perception.object_detector import ObjectDetector

    image = cv2.imread(str(FIXTURES / "corridor_doors.jpg"))
    detections = ObjectDetector().detect(image)
    assert [d for d in detections if "door" in d.class_name] == []
