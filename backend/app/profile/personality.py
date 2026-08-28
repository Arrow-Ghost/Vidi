"""
User profile and personality model — Parts 4, 5, 6, 15, 16.

PERSONALITY IS A VECTOR, NOT A SWITCH (Part 15). A user is not "an extrovert";
they are 0.8 explorer, 0.7 extrovert, 0.9 nature-lover, 0.4 chatterbox. Traits
combine, and the assistant's behaviour is derived from the whole vector rather
than a single label.

WHAT THIS IS NOT (Part 16): this is a PREFERENCE MODEL, not a psychological
assessment. Traits move only from things the user explicitly says. Nothing here
infers, diagnoses, or labels anyone from their behaviour, speech, or
appearance, and no clinical vocabulary exists anywhere in this module.

THE SAFETY BOUNDARY (Part 56): personality shapes style, proactivity, and
suggestions. It can never touch safety, privacy, identity confidence, or
consent. That is enforced structurally in AttentionManager — a trait can lower
the bar for chatter but can never raise the bar for a hazard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Trait(str, Enum):
    """The personality dimensions from Part 6. Users hold several at once."""

    EXTROVERT = "extrovert"
    INTROVERT = "introvert"
    EXPLORER = "explorer"
    ADVENTURER = "adventurer"
    SOCIAL = "social"
    CURIOUS = "curious"
    CALM = "calm"
    PRACTICAL = "practical"
    CHATTERBOX = "chatterbox"
    MINIMALIST = "minimalist"
    INDEPENDENT = "independent"
    CAUTIOUS = "cautious"
    NATURE_LOVER = "nature_lover"
    ROUTINE_ORIENTED = "routine_oriented"
    SPONTANEOUS = "spontaneous"
    ACHIEVEMENT_ORIENTED = "achievement_oriented"
    FAMILY_ORIENTED = "family_oriented"
    FRIEND_ORIENTED = "friend_oriented"


ALL_TRAITS = [t.value for t in Trait]

# Traits that pull in opposite directions. Raising one nudges the other down,
# so a user cannot end up strongly extrovert AND strongly introvert at once —
# which would make the derived behaviour incoherent.
OPPOSING = {
    Trait.EXTROVERT.value: Trait.INTROVERT.value,
    Trait.INTROVERT.value: Trait.EXTROVERT.value,
    Trait.CHATTERBOX.value: Trait.MINIMALIST.value,
    Trait.MINIMALIST.value: Trait.CHATTERBOX.value,
    Trait.CAUTIOUS.value: Trait.ADVENTURER.value,
    Trait.ADVENTURER.value: Trait.CAUTIOUS.value,
    Trait.ROUTINE_ORIENTED.value: Trait.SPONTANEOUS.value,
    Trait.SPONTANEOUS.value: Trait.ROUTINE_ORIENTED.value,
}

# How far an explicit statement moves a trait. Deliberately large enough that
# saying something once has a visible effect, small enough that one remark does
# not redefine the user.
EXPLICIT_STEP = 0.25
NEUTRAL = 0.5


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class PersonalityVector:
    """Trait name -> strength in 0..1. Absent traits are neutral (0.5)."""

    weights: Dict[str, float] = field(default_factory=dict)

    def get(self, trait: str) -> float:
        return self.weights.get(trait, NEUTRAL)

    def set(self, trait: str, value: float) -> None:
        if trait not in ALL_TRAITS:
            raise ValueError("unknown trait: " + str(trait))
        self.weights[trait] = clamp(value)

    def adjust(self, trait: str, delta: float) -> float:
        """
        Move a trait, and move its opposite the other way so the profile stays
        internally consistent.
        """
        if trait not in ALL_TRAITS:
            raise ValueError("unknown trait: " + str(trait))
        new_value = clamp(self.get(trait) + delta)
        self.weights[trait] = new_value

        opposite = OPPOSING.get(trait)
        if opposite is not None:
            self.weights[opposite] = clamp(self.get(opposite) - delta * 0.6)
        return new_value

    def dominant(self, limit: int = 3, floor: float = 0.6) -> List[str]:
        """The traits actually worth reasoning about."""
        strong = [(t, v) for t, v in self.weights.items() if v >= floor]
        strong.sort(key=lambda pair: pair[1], reverse=True)
        return [t for t, _ in strong[:limit]]

    def payload(self) -> Dict[str, float]:
        return {t: round(v, 2) for t, v in sorted(
            self.weights.items(), key=lambda p: p[1], reverse=True
        )}


@dataclass
class CommunicationStyle:
    """How much the assistant talks and how often it speaks first (Part 4)."""

    verbosity: str = "medium"        # low | medium | high
    proactive_level: str = "medium"  # low | medium | high
    humor: str = "light"             # none | light | playful
    encouragement: str = "medium"    # low | medium | high

    def payload(self) -> Dict[str, str]:
        return {
            "verbosity": self.verbosity,
            "proactive_level": self.proactive_level,
            "humor": self.humor,
            "encouragement": self.encouragement,
        }


@dataclass
class UserProfile:
    name: Optional[str] = None
    personality: PersonalityVector = field(default_factory=PersonalityVector)
    communication: CommunicationStyle = field(default_factory=CommunicationStyle)
    preferences: Dict[str, Any] = field(default_factory=dict)

    def payload(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "personality": self.personality.payload(),
            "dominant_traits": self.personality.dominant(),
            "communication": self.communication.payload(),
            "preferences": dict(self.preferences),
        }


# ---------------------------------------------------------------------------
# Behaviour derived from the vector
# ---------------------------------------------------------------------------

@dataclass
class BehaviourProfile:
    """
    Concrete knobs the rest of the system reads. Everything here is derived
    from the personality vector plus explicit communication settings — this is
    what makes personality real behaviour rather than a line in a prompt.
    """

    # 0..1. How strong an event must be before it is worth interrupting for.
    # HIGH means the user wants to be left alone.
    interruption_threshold: float = 0.5
    # Seconds between non-critical announcements.
    proactive_cooldown_s: float = 20.0
    # Whether to surface people the user knows, unprompted.
    announce_friends: bool = True
    # Whether to suggest going somewhere / meeting people.
    suggest_social: bool = True
    suggest_exploration: bool = True
    # How much optional scene commentary to add.
    environmental_detail: str = "medium"   # low | medium | high
    sentence_budget: int = 1               # sentences per spoken reply
    encouragement: str = "medium"

    def payload(self) -> Dict[str, Any]:
        return {
            "interruption_threshold": round(self.interruption_threshold, 2),
            "proactive_cooldown_s": round(self.proactive_cooldown_s, 1),
            "announce_friends": self.announce_friends,
            "suggest_social": self.suggest_social,
            "suggest_exploration": self.suggest_exploration,
            "environmental_detail": self.environmental_detail,
            "sentence_budget": self.sentence_budget,
            "encouragement": self.encouragement,
        }


_VERBOSITY_BUDGET = {"low": 1, "medium": 2, "high": 3}
_PROACTIVE_BASE = {"low": 0.72, "medium": 0.5, "high": 0.34}
_PROACTIVE_COOLDOWN = {"low": 45.0, "medium": 20.0, "high": 12.0}


class PersonalityEngine:
    """
    Turns a profile into behaviour (Part 5).

    Deliberately deterministic. The LLM is told the resulting STYLE, but the
    decisions about whether to interrupt, how often, and what kinds of
    suggestion are allowed are computed here — so behaviour is testable and
    cannot drift with model temperature.
    """

    def __init__(self, profile: Optional[UserProfile] = None) -> None:
        self.profile = profile or UserProfile()

    def personality_vector(self) -> PersonalityVector:
        return self.profile.personality

    def behaviour(self) -> BehaviourProfile:
        vector = self.profile.personality
        comms = self.profile.communication

        threshold = _PROACTIVE_BASE.get(comms.proactive_level, 0.5)
        cooldown = _PROACTIVE_COOLDOWN.get(comms.proactive_level, 20.0)

        # Talkative / minimalist traits pull the interruption bar around.
        threshold -= (vector.get(Trait.CHATTERBOX.value) - NEUTRAL) * 0.30
        threshold += (vector.get(Trait.MINIMALIST.value) - NEUTRAL) * 0.30
        threshold += (vector.get(Trait.INTROVERT.value) - NEUTRAL) * 0.20
        threshold -= (vector.get(Trait.EXTROVERT.value) - NEUTRAL) * 0.20
        # Calm users want fewer interruptions; independent users want fewer still.
        threshold += (vector.get(Trait.CALM.value) - NEUTRAL) * 0.15
        threshold += (vector.get(Trait.INDEPENDENT.value) - NEUTRAL) * 0.15

        cooldown *= 1.0 + (vector.get(Trait.MINIMALIST.value) - NEUTRAL) * 0.8
        cooldown *= 1.0 - (vector.get(Trait.CHATTERBOX.value) - NEUTRAL) * 0.6

        budget = _VERBOSITY_BUDGET.get(comms.verbosity, 2)
        if vector.get(Trait.PRACTICAL.value) > 0.7 or vector.get(Trait.MINIMALIST.value) > 0.7:
            budget = 1
        if vector.get(Trait.CHATTERBOX.value) > 0.7 or vector.get(Trait.CURIOUS.value) > 0.75:
            budget = max(budget, 3)

        detail = "medium"
        if vector.get(Trait.CURIOUS.value) > 0.65 or vector.get(Trait.EXPLORER.value) > 0.65:
            detail = "high"
        if vector.get(Trait.PRACTICAL.value) > 0.65 or vector.get(Trait.MINIMALIST.value) > 0.65:
            detail = "low"

        # Part 8: an introvert must not be pushed to socialize. Friends are
        # still *identified* — only the nudging stops.
        social = (
            vector.get(Trait.INTROVERT.value) < 0.65
            and self.profile.preferences.get("suggest_social", True) is not False
        )
        exploration = (
            max(vector.get(Trait.EXPLORER.value), vector.get(Trait.ADVENTURER.value),
                vector.get(Trait.SPONTANEOUS.value)) >= 0.55
            and self.profile.preferences.get("suggest_places", True) is not False
        )
        announce_friends = self.profile.preferences.get("announce_friends", True) is not False

        return BehaviourProfile(
            interruption_threshold=clamp(threshold, 0.15, 0.9),
            proactive_cooldown_s=clamp(cooldown, 6.0, 90.0),
            announce_friends=announce_friends,
            suggest_social=social,
            suggest_exploration=exploration,
            environmental_detail=detail,
            sentence_budget=budget,
            encouragement=comms.encouragement,
        )

    # ------------------------------------------------------------------
    # Style instructions handed to the LLM
    # ------------------------------------------------------------------

    def style_prompt(self) -> str:
        """
        A short, concrete description of HOW to talk to this user. Appended to
        the safety prompt, never replacing it — Part 56.
        """
        profile = self.profile
        vector = profile.personality
        behaviour = self.behaviour()
        lines: List[str] = []

        if profile.name:
            lines.append(
                "The user's name is " + profile.name + ". Use it occasionally, not every turn."
            )

        dominant = vector.dominant()
        if dominant:
            lines.append("Their stated preferences lean: " + ", ".join(dominant) + ".")

        if behaviour.sentence_budget <= 1:
            lines.append("Keep replies to ONE short sentence. Lead with the actionable part.")
        elif behaviour.sentence_budget == 2:
            lines.append("Keep replies to at most two short sentences.")
        else:
            lines.append(
                "You may use up to three sentences and add a little optional detail "
                "when it is genuinely interesting."
            )

        if vector.get(Trait.PRACTICAL.value) > 0.65:
            lines.append(
                "Be direct and functional: 'Door ahead, slightly right.' No commentary."
            )
        if vector.get(Trait.CALM.value) > 0.65:
            lines.append("Keep a calm, unhurried, reassuring tone.")
        if vector.get(Trait.CURIOUS.value) > 0.65:
            lines.append(
                "They enjoy context, so you may offer one extra optional detail and "
                "ask if they want to know more."
            )
        if profile.communication.humor == "playful":
            lines.append("A little warmth and light humour is welcome.")
        elif profile.communication.humor == "none":
            lines.append("Stay matter-of-fact; no jokes.")
        if behaviour.encouragement == "high":
            lines.append("Be encouraging and supportive when they are working through a task.")

        if not behaviour.suggest_social:
            lines.append(
                "Do NOT encourage them to approach or talk to people. You may still say "
                "who is nearby if asked or if they enabled friend alerts, phrased as an "
                "option: 'Ravi is nearby if you'd like to say hello.' Never push."
            )
        elif vector.get(Trait.EXTROVERT.value) > 0.65:
            lines.append(
                "They like knowing when people they know are around, so surfacing that "
                "is welcome."
            )

        if not behaviour.suggest_exploration:
            lines.append("Do not volunteer places to go unless they ask.")
        elif vector.get(Trait.NATURE_LOVER.value) > 0.65:
            lines.append(
                "They like green and outdoor spaces, so those are worth mentioning when "
                "genuinely detected."
            )

        lines.append(
            "These are style preferences only. They NEVER change what you report about "
            "safety, distance, identity, or confidence, and never allow you to state "
            "something perception did not actually establish."
        )
        return "\n".join("- " + line for line in lines)
