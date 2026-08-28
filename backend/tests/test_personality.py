"""
Personality, attention, recommendation and haptic tests — Parts 4-16, 33-44, 56.

The most important assertions here are NEGATIVE. Anyone can make an assistant
talk more; the hard part is proving that a preference can never silence a
hazard, that a trait can never invent a fact, and that "personality" never
becomes a diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.agent.attention import AttentionManager
from app.agent.recommendations import RecommendationEngine
from app.agent.tools import ToolRegistry
from app.events.bus import EventBus, EventType, Severity, WorldEvent
from app.haptics.engine import (
    COOLDOWN_S,
    HapticEngine,
    HapticEvent,
    HapticSettings,
    haptic_for_event,
)
from app.memory.store import MemoryStore
from app.perception.direction import DirectionBucket
from app.profile.personality import (
    PersonalityEngine,
    PersonalityVector,
    Trait,
    UserProfile,
)
from app.perception.world_model import WorldModel


def run(coro):
    import asyncio

    return asyncio.run(coro)


def profile(**traits) -> UserProfile:
    p = UserProfile()
    for trait, value in traits.items():
        p.personality.set(trait, value)
    return p


def event(kind=EventType.FRIEND_DETECTED, severity=Severity.SOCIAL,
          key="friend:Ravi", confidence=0.95, **data) -> WorldEvent:
    return WorldEvent(type=kind, severity=severity, key=key,
                      summary="x", confidence=confidence, data=data)


# ---------------- personality vector (Part 15) ----------------

def test_personality_is_a_vector_not_a_switch():
    vector = PersonalityVector()
    vector.set(Trait.EXPLORER.value, 0.8)
    vector.set(Trait.EXTROVERT.value, 0.7)
    vector.set(Trait.NATURE_LOVER.value, 0.9)
    assert vector.dominant() == ["nature_lover", "explorer", "extrovert"]
    # An unset trait is neutral, not absent-and-zero.
    assert vector.get(Trait.CAUTIOUS.value) == 0.5


def test_opposing_traits_stay_consistent():
    """A user cannot be strongly extrovert AND strongly introvert."""
    vector = PersonalityVector()
    vector.set(Trait.INTROVERT.value, 0.9)
    vector.adjust(Trait.EXTROVERT.value, 0.4)
    assert vector.get(Trait.INTROVERT.value) < 0.9


def test_unknown_traits_are_rejected():
    with pytest.raises(ValueError):
        PersonalityVector().set("telepathic", 0.9)


# ---------------- derived behaviour (Part 5) ----------------

def test_talkative_and_minimal_profiles_derive_opposite_behaviour():
    chatty = PersonalityEngine(profile(chatterbox=0.9, extrovert=0.85)).behaviour()
    terse = PersonalityEngine(profile(minimalist=0.9, practical=0.9)).behaviour()

    assert chatty.interruption_threshold < terse.interruption_threshold
    assert chatty.sentence_budget > terse.sentence_budget
    assert terse.environmental_detail == "low"


def test_introvert_is_not_pushed_to_socialize():
    """Part 8: friends are still identified; the nudging stops."""
    behaviour = PersonalityEngine(profile(introvert=0.85)).behaviour()
    assert behaviour.suggest_social is False
    assert behaviour.announce_friends is True


def test_style_prompt_states_the_safety_boundary():
    prompt = PersonalityEngine(profile(extrovert=0.9)).style_prompt()
    assert "NEVER change what you report about" in prompt


def test_style_prompt_tells_an_introvert_profile_not_to_push():
    prompt = PersonalityEngine(profile(introvert=0.85)).style_prompt()
    assert "Do NOT encourage them to approach" in prompt
    assert "if you'd like to say hello" in prompt


# ---------------- attention: personality changes behaviour (Part 33) ----------------

def test_same_event_reaches_one_user_and_not_another():
    friend = event()
    extrovert = AttentionManager(PersonalityEngine(profile(extrovert=0.9, social=0.85)))
    introvert = AttentionManager(PersonalityEngine(profile(introvert=0.9, minimalist=0.85)))

    assert extrovert.evaluate(friend).speak is True
    assert introvert.evaluate(friend).speak is False


def test_place_events_are_suppressed_for_a_practical_user():
    place = event(EventType.PLACE_DETECTED, Severity.AMBIENT, "place:park", subject="park")
    manager = AttentionManager(PersonalityEngine(profile(practical=0.9, minimalist=0.85)))
    assert manager.evaluate(place).speak is False


# ---------------- THE safety boundary (Part 56) ----------------

def test_no_preference_or_trait_can_silence_a_hazard():
    """
    Every suppression mechanism at once — quiet mode, alerts disabled, an
    extremely reserved profile, and the agent mid-sentence — must still not
    stop a safety event.
    """
    engine = PersonalityEngine(profile(minimalist=1.0, introvert=1.0, calm=1.0))
    engine.profile.preferences.update({
        "announce_friends": False, "announce_crowds": False,
        "suggest_places": False, "haptics_enabled": False,
    })
    engine.profile.communication.proactive_level = "low"

    manager = AttentionManager(engine)
    manager.quiet_mode = True

    hazard = event(EventType.OBSTACLE_DETECTED, Severity.SAFETY, "obstacle:1")
    decision = manager.evaluate(hazard, agent_is_busy=True)

    assert decision.speak is True
    assert decision.reason == "safety_override"


def test_quiet_mode_silences_ambient_but_not_navigation():
    manager = AttentionManager(PersonalityEngine(profile(explorer=0.9)))
    manager.quiet_mode = True
    assert manager.evaluate(
        event(EventType.PLACE_DETECTED, Severity.AMBIENT, "place:park")
    ).speak is False
    assert manager.evaluate(
        event(EventType.FRIEND_APPROACHING, Severity.NAVIGATION, "nav:1")
    ).speak is True


# ---------------- attention hygiene (Parts 36, 42) ----------------

def test_nothing_speaks_over_the_user():
    manager = AttentionManager(PersonalityEngine(profile(extrovert=0.9)))
    assert manager.evaluate(event(), agent_is_busy=True).reason == "agent_speaking"


def test_the_same_thing_is_not_announced_twice():
    manager = AttentionManager(PersonalityEngine(profile(extrovert=0.9)))
    assert manager.evaluate(event()).speak is True
    assert manager.evaluate(event()).reason == "already_announced"


def test_cooldown_applies_between_different_events():
    clock = {"t": 1000.0}
    manager = AttentionManager(
        PersonalityEngine(profile(extrovert=0.9)), clock=lambda: clock["t"]
    )
    assert manager.evaluate(event(key="friend:A")).speak is True
    assert manager.evaluate(event(key="friend:B")).reason == "cooldown"
    clock["t"] += 120
    assert manager.evaluate(event(key="friend:B")).speak is True


def test_low_confidence_events_score_lower():
    manager = AttentionManager(PersonalityEngine(profile(extrovert=0.8)))
    strong, _ = manager.score(event(confidence=0.98))
    weak, _ = manager.score(event(confidence=0.30))
    assert strong > weak


def test_disabling_friend_alerts_is_respected():
    engine = PersonalityEngine(profile(extrovert=0.9))
    engine.profile.preferences["announce_friends"] = False
    manager = AttentionManager(engine)
    assert manager.evaluate(event()).reason == "friend_alerts_disabled"


# ---------------- profile tools (Parts 4, 16) ----------------

@pytest.fixture()
def registry_and_engine(tmp_path):
    store = MemoryStore(db_path=tmp_path / "p.db")
    engine = PersonalityEngine(store.load_profile())
    registry = ToolRegistry(WorldModel(), store=store, personality=engine)
    yield registry, engine, store
    store.close()


def test_user_can_state_their_name_and_preferences(registry_and_engine):
    registry, engine, store = registry_and_engine
    run(registry.execute("set_name", {"name": "Shrish"}))
    run(registry.execute("set_personality", {"trait": "extrovert"}))
    run(registry.execute("set_personality", {"trait": "exploring"}))
    run(registry.execute("set_preference", {"key": "likes_nature", "enabled": True}))

    assert engine.profile.name == "Shrish"
    assert engine.profile.personality.get("extrovert") > 0.5
    assert engine.profile.personality.get("explorer") > 0.5
    # And it survives a restart.
    assert store.load_profile().name == "Shrish"


def test_be_less_talkative_actually_changes_behaviour(registry_and_engine):
    registry, engine, _ = registry_and_engine
    before = engine.behaviour()
    run(registry.execute("set_communication", {"verbosity": "low", "proactive_level": "low"}))
    after = engine.behaviour()
    assert after.sentence_budget <= before.sentence_budget
    assert after.interruption_threshold > before.interruption_threshold


def test_clinical_labels_are_refused_not_stored(registry_and_engine):
    """Part 16: this is a preference model and must never record a diagnosis."""
    registry, engine, _ = registry_and_engine
    result = run(registry.execute("set_personality", {"trait": "clinically depressed"}))
    assert result["saved"] is False
    assert result["refused"] is True
    assert engine.profile.personality.payload() == {}


def test_denying_a_trait_moves_it_down(registry_and_engine):
    registry, engine, _ = registry_and_engine
    run(registry.execute("set_personality", {"trait": "extrovert"}))
    high = engine.profile.personality.get("extrovert")
    run(registry.execute("set_personality", {"trait": "extrovert", "strength": "not"}))
    assert engine.profile.personality.get("extrovert") < high


def test_friend_groups_flag_members_without_a_face(registry_and_engine):
    registry, _, _ = registry_and_engine
    result = run(registry.execute(
        "create_friend_group", {"name": "college", "members": "Ravi, Ankit"}
    ))
    assert result["saved"] is True
    assert set(result["not_yet_enrolled"]) == {"Ravi", "Ankit"}


# ---------------- recommendations (Parts 9, 34, 35) ----------------

@dataclass
class FakeSign:
    text: str
    confidence: float = 0.95
    direction: object = DirectionBucket.CENTER


def test_recommendations_rank_by_stated_interest():
    nature = RecommendationEngine(PersonalityEngine(profile(nature_lover=0.95, explorer=0.8)))
    quiet = RecommendationEngine(PersonalityEngine(profile(introvert=0.9, curious=0.8)))
    signs = [FakeSign("City Park"), FakeSign("Central Library")]

    assert nature.recommend(signs, [])[0].place.category == "park"
    # This user has no explorer trait, so nothing is volunteered unprompted...
    assert quiet.recommend(signs, []) == []
    # ...but asking directly must still get an answer, ranked to their interests.
    assert quiet.recommend(signs, [], proactive=False)[0].place.category == "library"


def test_a_place_is_not_suggested_twice(registry_and_engine):
    engine = RecommendationEngine(PersonalityEngine(profile(explorer=0.9, nature_lover=0.9)))
    signs = [FakeSign("City Park")]
    first = engine.recommend(signs, [])
    assert first
    for item in first:
        engine.mark_mentioned(item.place)
    assert engine.recommend(signs, []) == []


def test_recommendations_never_invent_qualities():
    engine = RecommendationEngine(PersonalityEngine(profile(explorer=0.9, nature_lover=0.9)))
    text = engine.describe(engine.recommend([FakeSign("City Park")], [])[0]).lower()
    for word in ("beautiful", "lovely", "safe", "peaceful", "empty", "quiet and"):
        assert word not in text
    # And no fabricated measurement — there is no GPS in this system.
    assert "metre" not in text and "meter" not in text


def test_weak_scene_evidence_is_phrased_as_a_possibility():
    engine = RecommendationEngine(PersonalityEngine(profile(nature_lover=0.95)))

    @dataclass
    class E:
        type: str

    places = engine.detector.from_scene([E("bench"), E("potted plant")])
    assert places and places[0].evidence == "scene"
    assert places[0].confidence < 0.6


# ---------------- haptics (Parts 39-44) ----------------

def test_every_haptic_event_has_a_pattern():
    from app.haptics.engine import PATTERNS

    for event_name in HapticEvent:
        assert event_name in PATTERNS, "no pattern for " + event_name.value


def test_repeated_obstacle_does_not_buzz_every_frame():
    """Part 42: an object staying in view must not vibrate continuously."""
    clock = {"t": 100.0}
    engine = HapticEngine(clock=lambda: clock["t"])
    assert engine.trigger(HapticEvent.OBSTACLE_NEAR, "obs:1") is not None
    for _ in range(10):
        assert engine.trigger(HapticEvent.OBSTACLE_NEAR, "obs:1") is None
    clock["t"] += COOLDOWN_S[engine.trigger.__self__.settings and __import__(
        "app.haptics.engine", fromlist=["HapticPriority"]).HapticPriority.HIGH] + 1
    assert engine.trigger(HapticEvent.OBSTACLE_NEAR, "obs:1") is not None


def test_different_objects_are_tracked_separately():
    engine = HapticEngine()
    assert engine.trigger(HapticEvent.FRIEND_DETECTED, "friend:Ravi") is not None
    assert engine.trigger(HapticEvent.FRIEND_DETECTED, "friend:Ankit") is not None


def test_safety_only_mode_keeps_hazards_and_drops_the_rest():
    engine = HapticEngine(HapticSettings(safety_only=True))
    assert engine.trigger(HapticEvent.FRIEND_DETECTED, "f") is None
    assert engine.trigger(HapticEvent.OBSTACLE_CRITICAL, "o") is not None


def test_disabling_haptics_still_allows_safety():
    engine = HapticEngine(HapticSettings(enabled=False))
    assert engine.trigger(HapticEvent.CROWD_DETECTED, "c") is None
    assert engine.trigger(HapticEvent.OBSTACLE_CRITICAL, "o") is not None


def test_per_category_preferences(registry_and_engine):
    engine = HapticEngine()
    engine.preferences = {"haptics_friend": False}
    assert engine.trigger(HapticEvent.FRIEND_DETECTED, "f") is None
    assert engine.trigger(HapticEvent.CROWD_DETECTED, "c") is not None


def test_single_motor_provider_does_not_claim_direction():
    """Part 40: never claim directional haptics the hardware cannot deliver."""
    engine = HapticEngine()
    engine.set_provider("browser-vibrate", True, supports_direction=False)
    dispatch = engine.trigger(HapticEvent.NAVIGATION_LEFT, "nav")
    assert "cannot convey direction" in dispatch.note

    dual = HapticEngine()
    dual.set_provider("gamepad", True, supports_direction=True)
    assert dual.trigger(HapticEvent.NAVIGATION_LEFT, "nav").note == ""


def test_intensity_scales_the_pattern():
    quiet = HapticEngine(HapticSettings(intensity=0.4))
    loud = HapticEngine(HapticSettings(intensity=1.4))
    a = quiet.trigger(HapticEvent.FRIEND_DETECTED, "f").pattern.duration_ms
    b = loud.trigger(HapticEvent.FRIEND_DETECTED, "f").pattern.duration_ms
    assert a < b


def test_world_events_map_onto_haptics():
    assert haptic_for_event("FRIEND_DETECTED", "SOCIAL") == HapticEvent.FRIEND_DETECTED
    assert haptic_for_event("CROWD_APPROACHING", "NAVIGATION") == HapticEvent.CROWD_APPROACHING
    # Anything marked SAFETY becomes the critical pattern regardless of type.
    assert haptic_for_event("ANYTHING", "SAFETY") == HapticEvent.OBSTACLE_CRITICAL
    assert haptic_for_event("SIGN_DETECTED", "AMBIENT") is None


# ---------------- event bus (Part 48) ----------------

def test_event_history_is_bounded():
    bus = EventBus(history=10)
    for i in range(50):
        run(bus.publish(event(key="k" + str(i))))
    assert len(bus.history) == 10
    assert bus.stats()["total"] == 50


def test_one_failing_subscriber_does_not_stop_the_others():
    bus = EventBus()
    seen = []

    async def broken(_event):
        raise RuntimeError("boom")

    async def good(evt):
        seen.append(evt.key)

    bus.subscribe(broken)
    bus.subscribe(good)
    run(bus.publish(event(key="k1")))
    assert seen == ["k1"]
