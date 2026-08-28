"""
Tool layer tests — Segments 12, 31, 32, 33.

These encode the project's non-negotiable honesty rules as executable
assertions. Most of them are about what the system must REFUSE to claim:
absence, precision, and identity are the three things a monocular camera
cannot establish, and each has a test here.
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent import vocabulary
from app.agent.tools import ToolRegistry
from app.perception.direction import DirectionEstimate, DirectionBucket
from app.perception.distance import DistanceEstimate, DistanceRange
from app.perception.world_model import Entity, WorldModel


def _entity(
    entity_id: str,
    entity_type: str,
    bucket: DirectionBucket = DirectionBucket.RIGHT,
    distance: DistanceRange = DistanceRange.NEARBY,
    confidence: float = 0.9,
) -> Entity:
    return Entity(
        id=entity_id,
        type=entity_type,
        direction=DirectionEstimate(bucket=bucket, confidence=0.8),
        distance=DistanceEstimate(
            range=distance, confidence=0.5, method="test",
            label="approximately 2 to 4 meters away",
        ),
        confidence=confidence,
        is_person=entity_type == "person",
    )


def _registry(*entities: Entity) -> ToolRegistry:
    world = WorldModel()
    for entity in entities:
        world.entities[entity.id] = entity
    return ToolRegistry(world)


def run(coro):
    return asyncio.run(coro)


# ---------------- vocabulary ----------------

@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("backpack", ["backpack"]),
        ("my backpack", ["backpack"]),
        ("the phone", ["cell phone"]),
        ("water bottle", ["bottle"]),
        ("table", ["dining table"]),
        ("bottles", ["bottle"]),
        ("someone", ["person"]),
    ],
)
def test_spoken_words_resolve_to_detector_classes(spoken, expected):
    assert vocabulary.resolve(spoken) == expected


def test_bag_resolves_to_every_plausible_class():
    """One spoken word may honestly map to several classes — search them all."""
    assert set(vocabulary.resolve("bag")) == {"backpack", "handbag", "suitcase"}


def test_undetectable_things_resolve_to_nothing_rather_than_a_wrong_guess():
    for phrase in ["keys", "my wallet", "the door", "stairs"]:
        assert vocabulary.resolve(phrase) is None
        assert vocabulary.is_known_unsupported(phrase) is True


# ---------------- find_object ----------------

def test_find_object_returns_grounded_direction_and_distance():
    registry = _registry(_entity("backpack_1", "backpack"))
    result = run(registry.execute("find_object", {"object_name": "my backpack"}))
    assert result["found"] is True
    candidate = result["candidates"][0]
    assert candidate["entity_id"] == "backpack_1"
    assert candidate["direction_phrase"] == "to your right"
    assert candidate["distance_is_reliable"] is True


def test_absent_object_is_reported_as_not_visible_never_as_not_present():
    """The camera has a narrow field of view; 'I don't see it' != 'it isn't there'."""
    registry = _registry(_entity("chair_1", "chair"))
    result = run(registry.execute("find_object", {"object_name": "backpack"}))
    assert result["found"] is False
    assert result["supported"] is True
    message = result["message"].lower()
    assert "does not mean it is absent" in message
    assert "outside the field" in message


def test_undetectable_object_is_flagged_unsupported_not_absent():
    """
    Asking for keys must not produce 'I don't see your keys' — the user would
    hear that as 'your keys are gone'. It must say it cannot look for them.
    """
    registry = _registry(_entity("chair_1", "chair"))
    result = run(registry.execute("find_object", {"object_name": "my keys"}))
    assert result["found"] is False
    assert result["supported"] is False
    assert "cannot reliably look for" in result["message"]


def test_multiple_candidates_are_all_reported():
    registry = _registry(
        _entity("backpack_1", "backpack", bucket=DirectionBucket.LEFT, confidence=0.9),
        _entity("backpack_2", "backpack", bucket=DirectionBucket.CENTER, confidence=0.7),
    )
    result = run(registry.execute("find_object", {"object_name": "backpack"}))
    assert result["candidate_count"] == 2
    # Highest confidence first, so the agent leads with the best candidate.
    assert result["candidates"][0]["entity_id"] == "backpack_1"


def test_low_confidence_detection_is_marked_not_confident():
    registry = _registry(_entity("backpack_1", "backpack", confidence=0.48))
    result = run(registry.execute("find_object", {"object_name": "backpack"}))
    assert result["candidates"][0]["identification_is_confident"] is False


# ---------------- distance honesty ----------------

def test_unknown_distance_never_produces_a_phrase_to_speak():
    registry = _registry(_entity("bottle_1", "bottle", distance=DistanceRange.UNKNOWN))
    result = run(registry.execute("estimate_distance", {"entity_id": "bottle_1"}))
    assert result["distance_is_reliable"] is False
    assert result["distance_phrase"] is None
    assert "cannot reliably estimate" in result["message"]


def test_no_tool_result_ever_contains_a_precise_metric_number():
    """
    Structural guarantee behind 'never fabricate distance': the model can only
    speak values present in a tool result, and no tool result carries a
    metric figure it could quote.
    """
    registry = _registry(_entity("person_1", "person"), _entity("cup_1", "cup"))
    for name, args in [
        ("describe_scene", {}),
        ("find_object", {"object_name": "cup"}),
        ("estimate_distance", {"entity_id": "person_1"}),
        ("count_people", {}),
    ]:
        result = run(registry.execute(name, args))
        flattened = str(result)
        assert "meters away" not in flattened.replace(
            "approximately 2 to 4 meters away", ""
        ), name


# ---------------- lost tracks ----------------

def test_lost_entity_reports_lost_contact_rather_than_stale_data():
    registry = _registry()
    result = run(registry.execute("get_object_position", {"entity_id": "backpack_9"}))
    assert result["found"] is False
    assert "lost visual contact" in result["message"]


# ---------------- scene description ----------------

def test_empty_scene_is_not_reported_as_a_safe_or_empty_space():
    registry = _registry()
    result = run(registry.execute("describe_scene", {}))
    assert result["visible_entity_count"] == 0
    assert "does NOT mean the area is empty or safe" in result["note"]


def test_describe_scene_caps_output_so_the_user_is_not_flooded():
    """A blind user must never be read a list of twelve objects."""
    registry = _registry(*[_entity("chair_" + str(i), "chair") for i in range(12)])
    result = run(registry.execute("describe_scene", {}))
    assert len(result["entities"]) <= 5
    assert result["visible_entity_count"] == 12


def test_count_people_does_not_identify_anyone():
    registry = _registry(_entity("person_1", "person"), _entity("person_2", "person"))
    result = run(registry.execute("count_people", {}))
    assert result["person_count"] == 2
    assert "no identification" in result["note"].lower()
    for person in result["people"]:
        assert "name" not in person
        assert "identity" not in person


# ---------------- registry safety ----------------

def test_unknown_tool_is_refused_not_invented():
    registry = _registry()
    result = run(registry.execute("delete_all_files", {"path": "/"}))
    assert result["error"] == "unknown_tool"


def test_missing_and_unexpected_arguments_are_rejected():
    registry = _registry()
    assert run(registry.execute("find_object", {}))["error"] == "tool_error"
    bad = run(registry.execute("find_object", {"object_name": "cup", "rm": "-rf"}))
    assert bad["error"] == "tool_error"


def test_tool_failure_returns_a_structured_error_instead_of_raising():
    """A crashing tool must never kill the voice session mid-conversation."""
    registry = _registry()

    async def explode():
        raise RuntimeError("depth model segfaulted")

    registry._tools["describe_scene"].handler = explode
    result = run(registry.execute("describe_scene", {}))
    assert result["error"] == "internal_error"
    assert "segfaulted" in result["message"]


def test_slow_tool_times_out_rather_than_hanging_the_conversation():
    registry = _registry()

    async def hang():
        await asyncio.sleep(10)

    registry._tools["describe_scene"].handler = hang
    registry._tools["describe_scene"].timeout_s = 0.2
    result = run(registry.execute("describe_scene", {}))
    assert result["error"] == "timeout"


def test_every_call_is_recorded_in_the_developer_trace():
    registry = _registry(_entity("cup_1", "cup"))
    run(registry.execute("find_object", {"object_name": "cup"}))
    entry = registry.call_log[-1]
    assert entry["tool"] == "find_object"
    assert entry["latency_ms"] >= 0
    assert entry["result"]["found"] is True


def test_call_log_stays_bounded_over_a_long_session():
    registry = _registry()
    for _ in range(120):
        run(registry.execute("count_people", {}))
    assert len(registry.call_log) <= 50
