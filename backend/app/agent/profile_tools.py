"""
Profile, personality and preference tools — Parts 4, 16, 43, 52.

These are how "Remember I'm an extrovert", "be less talkative", and "tell me
when my friends are nearby" actually change the system's behaviour rather than
just being acknowledged politely.

TWO RULES ENCODED HERE:

1. Traits move ONLY from explicit statements (Part 16). There is no tool that
   infers personality from behaviour, and nothing in the system calls these
   handlers on its own — the model may only call them when the user says
   something about themselves.

2. No diagnosis. The trait vocabulary is a closed set of preference words. A
   request to record a clinical label is refused rather than stored, because
   this is a preference model and must never masquerade as an assessment.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.profile.personality import (
    ALL_TRAITS,
    EXPLICIT_STEP,
    PersonalityEngine,
    Trait,
    clamp,
)

logger = logging.getLogger("vidi.agent.profile_tools")

# Everyday phrasings mapped onto trait names, so the model does not have to
# guess our internal vocabulary.
TRAIT_SYNONYMS: Dict[str, str] = {
    "outgoing": "extrovert", "sociable": "extrovert", "people person": "extrovert",
    "shy": "introvert", "reserved": "introvert", "quiet": "introvert",
    "private": "introvert", "likes alone time": "introvert",
    "exploring": "explorer", "exploration": "explorer", "discovering": "explorer",
    "new places": "explorer", "wander": "explorer",
    "adventurous": "adventurer", "daring": "adventurer", "thrill": "adventurer",
    "friendly": "social", "meeting people": "social",
    "inquisitive": "curious", "asks questions": "curious", "learning": "curious",
    "relaxed": "calm", "chill": "calm", "peaceful": "calm",
    "efficient": "practical", "to the point": "practical", "no nonsense": "practical",
    "talkative": "chatterbox", "chatty": "chatterbox",
    "brief": "minimalist", "concise": "minimalist", "short answers": "minimalist",
    "self reliant": "independent", "independent": "independent",
    "careful": "cautious", "worried": "cautious", "safety first": "cautious",
    "nature": "nature_lover", "outdoors": "nature_lover", "parks": "nature_lover",
    "green spaces": "nature_lover", "gardens": "nature_lover",
    "routine": "routine_oriented", "schedule": "routine_oriented",
    "organised": "routine_oriented", "organized": "routine_oriented",
    "spontaneous": "spontaneous", "last minute": "spontaneous",
    "driven": "achievement_oriented", "ambitious": "achievement_oriented",
    "goals": "achievement_oriented",
    "family": "family_oriented", "friends": "friend_oriented",
}

# Words that indicate a clinical/diagnostic framing. These are refused.
CLINICAL_TERMS = {
    "depressed", "depression", "bipolar", "borderline", "schizophrenia",
    "schizophrenic", "adhd", "autistic", "autism", "ocd", "ptsd", "anxiety disorder",
    "personality disorder", "psychotic", "manic", "neurotic", "diagnosed",
    "mental illness", "disorder",
}

LEVELS = {"low", "medium", "high"}


def resolve_trait(text: str) -> Optional[str]:
    """Map an everyday word onto a trait name, or None if it is not one."""
    if not text:
        return None
    key = text.strip().lower().replace("-", " ").replace("_", " ")
    if key.replace(" ", "_") in ALL_TRAITS:
        return key.replace(" ", "_")
    if key in TRAIT_SYNONYMS:
        return TRAIT_SYNONYMS[key]
    for phrase, trait in TRAIT_SYNONYMS.items():
        if phrase in key:
            return trait
    for trait in ALL_TRAITS:
        if trait.replace("_", " ") in key:
            return trait
    return None


def looks_clinical(text: str) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in CLINICAL_TERMS)


def register_profile_tools(registry, store, engine: PersonalityEngine) -> None:
    """Attach profile tools. `engine` holds the live profile the agent uses."""
    from app.agent.tools import ToolSpec

    def persist() -> None:
        store.save_profile(engine.profile)

    async def set_name(name: str) -> Dict[str, Any]:
        engine.profile.name = name.strip()
        persist()
        return {"saved": True, "name": engine.profile.name}

    async def set_personality(trait: str, strength: str = "strong") -> Dict[str, Any]:
        """Record something the user said about themselves."""
        if looks_clinical(trait):
            return {
                "saved": False,
                "refused": True,
                "message": (
                    "That sounds like a clinical description, and this profile is only a "
                    "preference model — it must not record diagnoses. Tell the user you "
                    "can note how they like to be talked to instead, and suggest a "
                    "preference wording."
                ),
            }
        resolved = resolve_trait(trait)
        if resolved is None:
            return {
                "saved": False,
                "message": "No preference trait matches " + repr(trait) + ".",
                "available_traits": ALL_TRAITS,
            }

        step = {"slight": EXPLICIT_STEP / 2, "strong": EXPLICIT_STEP,
                "very": EXPLICIT_STEP * 1.5}.get(strength, EXPLICIT_STEP)
        # "not an extrovert" / "less of an explorer"
        if strength in ("not", "no", "negative"):
            step = -EXPLICIT_STEP

        value = engine.profile.personality.adjust(resolved, step)
        persist()
        return {
            "saved": True,
            "trait": resolved,
            "value": round(value, 2),
            "personality": engine.profile.personality.payload(),
            "dominant_traits": engine.profile.personality.dominant(),
        }

    async def set_communication(
        verbosity: str = "", proactive_level: str = "",
        humor: str = "", encouragement: str = "",
    ) -> Dict[str, Any]:
        """'Be more talkative', 'don't interrupt me so much', etc."""
        comms = engine.profile.communication
        changed = []
        for field_name, value in (
            ("verbosity", verbosity), ("proactive_level", proactive_level),
            ("encouragement", encouragement),
        ):
            if value:
                if value not in LEVELS:
                    return {"saved": False,
                            "message": field_name + " must be low, medium or high."}
                setattr(comms, field_name, value)
                changed.append(field_name)
        if humor:
            if humor not in {"none", "light", "playful"}:
                return {"saved": False, "message": "humor must be none, light or playful."}
            comms.humor = humor
            changed.append("humor")

        if not changed:
            return {"saved": False, "message": "Nothing to change."}
        persist()
        return {
            "saved": True,
            "changed": changed,
            "communication": comms.payload(),
            "behaviour": engine.behaviour().payload(),
        }

    async def set_preference(key: str, enabled: bool = True) -> Dict[str, Any]:
        """
        Toggle a behavioural preference, e.g. announce_friends,
        suggest_places, suggest_social, haptics_friend, haptics_crowd.
        """
        clean = key.strip().lower().replace(" ", "_")
        if looks_clinical(clean):
            return {"saved": False, "refused": True,
                    "message": "That is a clinical description and will not be stored."}
        engine.profile.preferences[clean] = bool(enabled)
        persist()
        return {
            "saved": True,
            "preference": clean,
            "enabled": bool(enabled),
            "behaviour": engine.behaviour().payload(),
            "scope": (
                "This preference controls what you volunteer WITHOUT being asked. It is "
                "NOT a refusal to answer. If the user directly asks a question this "
                "preference relates to, you MUST still call the relevant tool and answer "
                "them. Never tell the user you are unable to answer because of a "
                "preference they set."
            ),
        }

    async def get_profile() -> Dict[str, Any]:
        return {
            **engine.profile.payload(),
            "behaviour": engine.behaviour().payload(),
            "note": (
                "This is a preference model the user set explicitly. It is not an "
                "assessment. Describe it as preferences, never as a personality test "
                "result or diagnosis."
            ),
        }

    async def reset_profile() -> Dict[str, Any]:
        from app.profile.personality import UserProfile

        store.reset_profile()
        engine.profile = UserProfile()
        return {"reset": True, "message": "Cleared the profile and all personality settings."}

    # ---------------- friend groups (Part 52) ----------------

    async def create_friend_group(name: str, members: str) -> Dict[str, Any]:
        parsed = [m.strip() for m in members.split(",") if m.strip()]
        if not parsed:
            return {"saved": False, "message": "No members given."}
        enrolled = {r.name.lower() for r in store.list_enrollments("person")}
        unknown = [m for m in parsed if m.lower() not in enrolled]
        store.save_friend_group(name, parsed)
        return {
            "saved": True,
            "group": name,
            "members": parsed,
            "not_yet_enrolled": unknown,
            "note": (
                "Members under not_yet_enrolled have no face enrolled, so they cannot be "
                "recognized yet. Tell the user they need to add a photo for those."
                if unknown else ""
            ),
        }

    async def list_friend_groups() -> Dict[str, Any]:
        groups = store.list_friend_groups()
        return {"count": len(groups), "groups": groups}

    # ---------------- registration ----------------

    registry.register(ToolSpec(
        name="set_name",
        description=(
            "Record the user's name when they tell you it, e.g. 'remember my name is "
            "Shrish'."
        ),
        parameters={"type": "OBJECT", "properties": {"name": {"type": "STRING"}},
                    "required": ["name"]},
        handler=set_name,
    ))
    registry.register(ToolSpec(
        name="set_personality",
        description=(
            "Record something the user says about their own preferences or temperament, "
            "e.g. 'I'm an extrovert', 'I like exploring new places', 'I prefer quiet'. "
            "Call this ONLY when the user states it themselves — never infer it from how "
            "they talk or behave. strength may be 'slight', 'strong', 'very', or 'not' "
            "when they are denying a trait."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "trait": {"type": "STRING",
                          "description": "e.g. extrovert, introvert, explorer, nature_lover."},
                "strength": {"type": "STRING", "description": "slight | strong | very | not"},
            },
            "required": ["trait"],
        },
        handler=set_personality,
    ))
    registry.register(ToolSpec(
        name="set_communication",
        description=(
            "Change how the assistant talks. Use for 'be more talkative' "
            "(verbosity=high), 'be less talkative' (verbosity=low), 'don't interrupt me "
            "so much' (proactive_level=low), 'be more proactive' (proactive_level=high)."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "verbosity": {"type": "STRING", "description": "low | medium | high"},
                "proactive_level": {"type": "STRING", "description": "low | medium | high"},
                "humor": {"type": "STRING", "description": "none | light | playful"},
                "encouragement": {"type": "STRING", "description": "low | medium | high"},
            },
        },
        handler=set_communication,
    ))
    registry.register(ToolSpec(
        name="set_preference",
        description=(
            "Turn a behaviour on or off. Keys include announce_friends ('tell me when my "
            "friends are nearby'), suggest_places, suggest_social ('don't push me to talk "
            "to people'), haptics_friend, haptics_crowd, quiet_mode."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "key": {"type": "STRING"},
                "enabled": {"type": "BOOLEAN"},
            },
            "required": ["key"],
        },
        handler=set_preference,
    ))
    registry.register(ToolSpec(
        name="get_profile",
        description=(
            "Read back the user's stored name, preferences and communication settings. "
            "Use for 'what do you know about me', 'what are my settings'."
        ),
        parameters={"type": "OBJECT", "properties": {}},
        handler=get_profile,
    ))
    registry.register(ToolSpec(
        name="reset_profile",
        description="Erase the stored profile and personality settings when the user asks.",
        parameters={"type": "OBJECT", "properties": {}},
        handler=reset_profile,
    ))
    registry.register(ToolSpec(
        name="create_friend_group",
        description=(
            "Group enrolled people, e.g. 'Ravi, Ankit and Neha are my college friends'. "
            "members is a comma-separated list of names."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {"name": {"type": "STRING"}, "members": {"type": "STRING"}},
            "required": ["name", "members"],
        },
        handler=create_friend_group,
    ))
    registry.register(ToolSpec(
        name="list_friend_groups",
        description="List the user's friend groups and who is in them.",
        parameters={"type": "OBJECT", "properties": {}},
        handler=list_friend_groups,
    ))
