"""
Memory, tasks, and preparation tests — Segments 16, 17, 18, 34.

Runs entirely offline: the embedding client is left unconfigured so the store
exercises its keyword fallback. That is deliberate — the fallback is the path
that runs whenever the network is down, and it must be correct, not merely
present. A separate opt-in test covers real semantic search.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from app.agent.tools import ToolRegistry
from app.memory.embeddings import EmbeddingClient, cosine_similarity
from app.memory.store import MemoryStore
from app.perception.direction import DirectionBucket, DirectionEstimate
from app.perception.distance import DistanceEstimate, DistanceRange
from app.perception.world_model import Entity, WorldModel


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def store(tmp_path):
    memory_store = MemoryStore(db_path=tmp_path / "test.db")
    yield memory_store
    memory_store.close()


def _entity(entity_id: str, entity_type: str, confidence: float = 0.9) -> Entity:
    return Entity(
        id=entity_id,
        type=entity_type,
        direction=DirectionEstimate(bucket=DirectionBucket.RIGHT, confidence=0.8),
        distance=DistanceEstimate(DistanceRange.NEARBY, 0.6, "test", "about 2 to 4 meters away"),
        confidence=confidence,
        is_person=entity_type == "person",
    )


def _registry(store: MemoryStore, *entities: Entity) -> ToolRegistry:
    world = WorldModel()
    for entity in entities:
        world.entities[entity.id] = entity
    return ToolRegistry(world, store=store)


# ---------------- persistence ----------------

def test_memory_persists_across_store_restarts(tmp_path):
    """What the user asked to remember must survive a process restart."""
    path = tmp_path / "persist.db"
    first = MemoryStore(db_path=path)
    run(first.remember("object", "backpack", "my backpack is blue with a red zip"))
    first.close()

    second = MemoryStore(db_path=path)
    recovered = second.get_by_subject("backpack")
    assert recovered is not None
    assert "blue" in recovered.content
    second.close()


def test_remembering_the_same_subject_updates_rather_than_duplicates(store):
    run(store.remember("object", "backpack", "blue"))
    run(store.remember("object", "backpack", "actually it is green"))
    memories = store.list_memories()
    assert len(memories) == 1
    assert "green" in memories[0].content


def test_subjects_are_normalized_so_recall_is_not_case_sensitive(store):
    run(store.remember("place", "  My Classroom  ", "Room 204"))
    assert store.get_by_subject("my classroom") is not None


# ---------------- search ----------------

def test_keyword_fallback_finds_memories_without_embeddings(store):
    """The offline path must actually work, not just not crash."""
    assert store.embeddings.available is False
    run(store.remember("place", "classroom", "my classroom is Room 204"))
    run(store.remember("object", "backpack", "blue backpack"))

    results = run(store.search("which classroom am I in"))
    assert results
    assert results[0].subject == "classroom"


def test_search_returns_nothing_rather_than_a_bad_guess(store):
    run(store.remember("object", "backpack", "blue backpack"))
    assert run(store.search("quantum chromodynamics")) == []


@pytest.mark.skipif(
    os.environ.get("VIDI_LIVE_TESTS") != "1" or not os.environ.get("GEMINI_API_KEY"),
    reason="set VIDI_LIVE_TESTS=1 and GEMINI_API_KEY for real semantic search",
)
def test_semantic_search_beats_keyword_matching(tmp_path):
    """
    The real value of embeddings: finding a memory that shares NO words with
    the query. Keyword search cannot do this.
    """
    store = MemoryStore(
        db_path=tmp_path / "sem.db",
        embeddings=EmbeddingClient(os.environ["GEMINI_API_KEY"]),
    )
    try:
        run(store.remember("place", "classroom", "Room 204 in the science building"))
        run(store.remember("object", "backpack", "navy rucksack with a broken strap"))

        results = run(store.search("where do I have my lecture"))
        assert results and results[0].subject == "classroom"

        results = run(store.search("my bag"))
        assert results and results[0].subject == "backpack"
    finally:
        store.close()


def test_embeddings_degrade_to_none_instead_of_raising():
    client = EmbeddingClient(api_key=None)
    assert client.available is False
    assert run(client.embed("anything")) is None


def test_cosine_similarity_handles_mismatched_and_empty_vectors():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# ---------------- deletion / privacy ----------------

def test_forget_reports_the_true_count_and_really_deletes(store):
    run(store.remember("object", "backpack", "blue"))
    assert store.forget(subject="backpack") == 1
    assert store.get_by_subject("backpack") is None
    # Deleting something that was never there must report 0, not success.
    assert store.forget(subject="backpack") == 0


def test_forget_tool_does_not_claim_success_when_nothing_was_deleted(store):
    registry = _registry(store)
    result = run(registry.execute("forget", {"subject": "unicorn"}))
    assert result["deleted_count"] == 0
    assert "nothing saved" in result["message"].lower()
    assert "nothing was deleted" in result["message"].lower()


def test_forget_all_clears_everything(store):
    for i in range(3):
        run(store.remember("fact", "thing" + str(i), "value"))
    assert store.forget_all() == 3
    assert store.list_memories() == []


# ---------------- memory tools ----------------

def test_remember_and_recall_round_trip_through_tools(store):
    registry = _registry(store)
    saved = run(registry.execute(
        "remember",
        {"subject": "classroom", "content": "Room 204", "kind": "place"},
    ))
    assert saved["saved"] is True

    recalled = run(registry.execute("recall", {"query": "classroom"}))
    assert recalled["found"] is True
    assert "Room 204" in recalled["memories"][0]["content"]


def test_recall_miss_explains_that_only_explicit_memories_exist(store):
    registry = _registry(store)
    result = run(registry.execute("recall", {"query": "my car"}))
    assert result["found"] is False
    assert "explicitly" in result["message"]


def test_invalid_memory_kind_is_coerced_not_rejected(store):
    registry = _registry(store)
    result = run(registry.execute(
        "remember", {"subject": "x", "content": "y", "kind": "nonsense"}
    ))
    assert result["kind"] == "fact"


# ---------------- Segment 34: memory-aware finding ----------------

def test_saved_description_is_surfaced_when_candidates_are_ambiguous(store):
    """
    Two backpacks in view and the user once said theirs is blue: the tool
    surfaces that description — but explicitly must not claim a match.
    """
    run(store.remember("object", "backpack", "mine is blue with a red zip"))
    registry = _registry(
        store, _entity("backpack_1", "backpack", 0.9), _entity("backpack_2", "backpack", 0.8)
    )
    result = run(registry.execute("find_object", {"object_name": "backpack"}))
    assert result["candidate_count"] == 2
    assert "blue" in result["remembered_description"]
    assert "do NOT claim to have matched it" in result["disambiguation_note"]


def test_no_description_is_surfaced_when_there_is_only_one_candidate(store):
    run(store.remember("object", "backpack", "mine is blue"))
    registry = _registry(store, _entity("backpack_1", "backpack"))
    result = run(registry.execute("find_object", {"object_name": "backpack"}))
    assert "remembered_description" not in result


# ---------------- Segment 17: tasks ----------------

def test_task_lifecycle(store):
    registry = _registry(store)
    created = run(registry.execute(
        "create_task",
        {"title": "college", "due_iso": "2026-08-29T10:00:00", "due_text": "10 tomorrow"},
    ))
    assert created["created"] is True

    listed = run(registry.execute("get_tasks", {}))
    assert listed["count"] == 1
    assert listed["tasks"][0]["due"] == "10 tomorrow"

    done = run(registry.execute("complete_task", {"title": "college"}))
    assert done["completed"] is True
    assert run(registry.execute("get_tasks", {}))["count"] == 0


def test_completing_a_nonexistent_task_is_reported_honestly(store):
    registry = _registry(store)
    result = run(registry.execute("complete_task", {"title": "wash the dragon"}))
    assert result["completed"] is False
    assert "No pending task" in result["message"]


def test_unparseable_due_date_stores_the_task_without_inventing_a_time(store):
    """Better to hold a task with no time than to schedule it for the wrong day."""
    registry = _registry(store)
    run(registry.execute(
        "create_task", {"title": "dentist", "due_iso": "next tuesday-ish", "due_text": "next Tuesday"}
    ))
    task = store.list_tasks()[0]
    assert task.due_at is None
    assert task.due_text == "next Tuesday"


def test_empty_task_list_says_so(store):
    registry = _registry(store)
    assert run(registry.execute("get_tasks", {}))["count"] == 0


# ---------------- Segment 18: preparation mode ----------------

def test_checklist_separates_seen_from_not_seen_from_uncheckable(store):
    """
    The three categories must stay distinct. Collapsing 'not visible' into
    'missing' is the exact failure this system must never make.
    """
    registry = _registry(
        store, _entity("backpack_1", "backpack"), _entity("cell_phone_1", "cell phone")
    )
    result = run(registry.execute("check_checklist", {"name": "college"}))

    assert result["found"] is True
    seen_items = {s["item"] for s in result["seen"]}
    assert "backpack" in seen_items and "phone" in seen_items
    # Present in the list, not in view.
    assert "water bottle" in result["not_currently_visible"]
    # No detector category exists for these at all.
    assert "keys" in result["cannot_be_checked_visually"]
    assert "wallet" in result["cannot_be_checked_visually"]
    assert "NEVER say it is missing" in result["note"]


def test_checklist_supplies_correct_phrasing_for_each_category():
    """
    Regression test for a real bug: the agent conflated "not currently
    visible" with "cannot be checked", telling the user it could not check an
    item that was merely out of frame. Ready-made phrasing keeps them apart.
    """
    store = MemoryStore(db_path=":memory:")
    try:
        registry = _registry(store, _entity("backpack_1", "backpack"))
        phrasing = run(registry.execute("check_checklist", {"name": "college"}))["suggested_phrasing"]

        assert "I can see your backpack" in phrasing
        # Out of view -> "can't see", never "missing", never "can't check".
        assert "I can't see your" in phrasing and "water bottle" in phrasing
        # No detector at all -> "no way to check".
        assert "I have no way to check your" in phrasing
        assert "missing" not in phrasing.lower()
        assert "you don't have" not in phrasing.lower()
    finally:
        store.close()


def test_spoken_list_joining_reads_naturally():
    from app.agent.memory_tools import _join

    assert _join(["keys"]) == "keys"
    assert _join(["keys", "wallet"]) == "keys and wallet"
    assert _join(["keys", "wallet", "phone"]) == "keys, wallet and phone"
    assert _join([]) == ""


def test_unknown_checklist_lists_the_available_ones(store):
    registry = _registry(store)
    result = run(registry.execute("check_checklist", {"name": "spacewalk"}))
    assert result["found"] is False
    assert "college" in result["available"]


def test_custom_checklist_can_be_saved_and_used(store):
    registry = _registry(store, _entity("laptop_1", "laptop"))
    run(registry.execute("save_checklist", {"name": "studio", "items": "laptop, cup, chair"}))
    result = run(registry.execute("check_checklist", {"name": "studio"}))
    assert {s["item"] for s in result["seen"]} == {"laptop"}
    assert set(result["not_currently_visible"]) == {"cup", "chair"}


# ---------------- registry composition ----------------

def test_memory_tools_are_absent_without_a_store():
    """The perception-only path must not require a database to exist."""
    registry = ToolRegistry(WorldModel())
    assert "remember" not in registry.names()
    assert "find_object" in registry.names()


def test_memory_tools_are_present_with_a_store(store):
    registry = _registry(store)
    for name in ["remember", "recall", "forget", "create_task", "get_tasks", "check_checklist"]:
        assert name in registry.names()
