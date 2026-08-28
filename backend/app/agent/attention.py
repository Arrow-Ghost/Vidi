"""
Relevance and attention — Parts 36, 37, 49, 56.

This is the module that decides whether the assistant opens its mouth.

For a blind user, speech is the only output channel and it cannot be skimmed
past. An assistant that narrates every crowd shift and every park is worse
than one that stays quiet, so the default answer here is silence and an event
has to earn its way through.

SCORING (Part 36). Each event gets:
    importance      from its severity class
    confidence      from perception
    novelty         has this key been announced recently?
    personal        does it match the user's stated preferences?
    personality     does their profile want this kind of thing at all?
then shouldSpeak() compares the total against a personality-derived threshold.

THE SAFETY BOUNDARY (Part 56), enforced structurally rather than by prompt:
SAFETY events bypass scoring, cooldowns, quiet mode, and the busy check
entirely. There is no profile setting, preference, or trait that can silence a
hazard — the early-return happens before any personality value is read.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.events.bus import EventType, Severity, WorldEvent
from app.profile.personality import PersonalityEngine, Trait

logger = logging.getLogger("vidi.agent.attention")

# Base importance per severity class.
_IMPORTANCE = {
    Severity.SAFETY: 1.00,
    Severity.NAVIGATION: 0.80,
    Severity.SOCIAL: 0.62,
    Severity.ENVIRONMENT: 0.45,
    Severity.AMBIENT: 0.30,
}

# Which traits make a given event class more or less wanted (Part 33).
_TRAIT_AFFINITY: Dict[EventType, Dict[str, float]] = {
    EventType.FRIEND_DETECTED: {
        Trait.EXTROVERT.value: 0.25, Trait.SOCIAL.value: 0.20,
        Trait.FRIEND_ORIENTED.value: 0.25, Trait.INTROVERT.value: -0.10,
    },
    EventType.FRIEND_DETECTED_IN_CROWD: {
        Trait.EXTROVERT.value: 0.25, Trait.FRIEND_ORIENTED.value: 0.25,
    },
    EventType.FRIEND_APPROACHING: {
        Trait.EXTROVERT.value: 0.20, Trait.FRIEND_ORIENTED.value: 0.20,
    },
    EventType.PLACE_DETECTED: {
        Trait.EXPLORER.value: 0.30, Trait.ADVENTURER.value: 0.20,
        Trait.NATURE_LOVER.value: 0.25, Trait.CURIOUS.value: 0.15,
        Trait.PRACTICAL.value: -0.20, Trait.MINIMALIST.value: -0.25,
    },
    EventType.RECOMMENDATION: {
        Trait.EXPLORER.value: 0.30, Trait.SPONTANEOUS.value: 0.20,
        Trait.PRACTICAL.value: -0.25, Trait.MINIMALIST.value: -0.30,
        Trait.ROUTINE_ORIENTED.value: -0.15,
    },
    EventType.CROWD_APPEARED: {
        Trait.CAUTIOUS.value: 0.25, Trait.INTROVERT.value: 0.15,
        Trait.MINIMALIST.value: -0.15,
    },
    EventType.CROWD_APPROACHING: {Trait.CAUTIOUS.value: 0.25},
    EventType.CROWD_CONGESTION: {Trait.CAUTIOUS.value: 0.20},
    EventType.TASK_DUE: {
        Trait.ROUTINE_ORIENTED.value: 0.25, Trait.ACHIEVEMENT_ORIENTED.value: 0.15,
    },
}

# Per-event-class minimum gap, before personality scaling.
_BASE_COOLDOWN_S: Dict[Severity, float] = {
    Severity.SAFETY: 0.0,
    Severity.NAVIGATION: 6.0,
    Severity.SOCIAL: 15.0,
    Severity.ENVIRONMENT: 25.0,
    Severity.AMBIENT: 90.0,
}

# How long an announced key stays "not novel".
NOVELTY_WINDOW_S = 120.0


@dataclass
class Decision:
    speak: bool
    haptic: bool
    score: float
    threshold: float
    reason: str
    event: WorldEvent

    def payload(self) -> Dict[str, Any]:
        return {
            "speak": self.speak,
            "haptic": self.haptic,
            "score": round(self.score, 3),
            "threshold": round(self.threshold, 3),
            "reason": self.reason,
            "event": self.event.payload(),
        }


@dataclass
class AttentionState:
    announced_at: Dict[str, float] = field(default_factory=dict)
    last_spoke_at: float = 0.0
    spoken: int = 0
    suppressed: int = 0
    suppression_reasons: Dict[str, int] = field(default_factory=dict)


class AttentionManager:
    """
    Deterministic gatekeeper between the event bus and the voice.

    Generalizes the earlier signage-only ProactiveAnnouncer to every event
    class, and adds the personality and personal-relevance terms the spec asks
    for. The signage behaviour it replaces is preserved: warnings jump the
    queue, a thing is announced once, and nothing speaks over a live turn.
    """

    def __init__(
        self,
        personality: Optional[PersonalityEngine] = None,
        clock=time.time,
    ) -> None:
        self.personality = personality or PersonalityEngine()
        self._clock = clock
        self.state = AttentionState()
        self.quiet_mode = False
        self.recent_decisions: List[Decision] = []

    # ------------------------------------------------------------------

    def evaluate(
        self, event: WorldEvent, agent_is_busy: bool = False
    ) -> Decision:
        now = self._clock()
        behaviour = self.personality.behaviour()

        # ---- SAFETY BYPASS (Part 56) --------------------------------
        # Deliberately before every personality, preference, cooldown and
        # quiet-mode check. Nothing a user configures can reach this path.
        if event.severity == Severity.SAFETY:
            return self._commit(
                Decision(True, True, 1.0, 0.0, "safety_override", event), now
            )

        if agent_is_busy:
            return self._reject(event, "agent_speaking", now)

        if self.quiet_mode and event.severity.value > Severity.NAVIGATION.value:
            return self._reject(event, "quiet_mode", now)

        # ---- preference switches ------------------------------------
        preferences = self.personality.profile.preferences
        if event.type.value.startswith("FRIEND") and not behaviour.announce_friends:
            return self._reject(event, "friend_alerts_disabled", now)
        if event.type in (EventType.PLACE_DETECTED, EventType.RECOMMENDATION):
            if not behaviour.suggest_exploration:
                return self._reject(event, "place_suggestions_disabled", now)
        if event.type.value.startswith("CROWD") and preferences.get("announce_crowds") is False:
            return self._reject(event, "crowd_alerts_disabled", now)

        # ---- novelty -------------------------------------------------
        last = self.state.announced_at.get(event.key)
        if last is not None and now - last < NOVELTY_WINDOW_S:
            return self._reject(event, "already_announced", now)

        # ---- cooldown ------------------------------------------------
        cooldown = _BASE_COOLDOWN_S[event.severity]
        cooldown *= behaviour.proactive_cooldown_s / 20.0  # 20s is the neutral baseline
        if now - self.state.last_spoke_at < cooldown:
            return self._reject(event, "cooldown", now)

        # ---- score ---------------------------------------------------
        score, threshold = self.score(event, behaviour)
        if score < threshold:
            return self._reject(event, "below_threshold", now, score, threshold)

        return self._commit(
            Decision(True, self._wants_haptic(event), score, threshold, "accepted", event),
            now,
        )

    def score(self, event: WorldEvent, behaviour=None) -> tuple:
        """Exposed separately so the dashboard can show the arithmetic."""
        behaviour = behaviour or self.personality.behaviour()
        vector = self.personality.personality_vector()

        importance = _IMPORTANCE[event.severity]
        # Low-confidence perception must not drive unprompted speech.
        confidence = max(0.0, min(1.0, event.confidence))

        affinity = 0.0
        for trait, weight in _TRAIT_AFFINITY.get(event.type, {}).items():
            affinity += (vector.get(trait) - 0.5) * 2.0 * weight

        personal = 0.0
        subject = str(event.data.get("subject", "")).lower()
        for key, value in self.personality.profile.preferences.items():
            if value and key.startswith("likes_") and key[6:] and key[6:] in subject:
                personal += 0.15

        score = importance * (0.55 + 0.45 * confidence) + affinity + personal
        return max(0.0, score), behaviour.interruption_threshold

    # ------------------------------------------------------------------

    def _wants_haptic(self, event: WorldEvent) -> bool:
        preferences = self.personality.profile.preferences
        if preferences.get("haptics_enabled") is False:
            return False
        if event.type.value.startswith("FRIEND"):
            return preferences.get("haptics_friend", True) is not False
        if event.type.value.startswith("CROWD"):
            return preferences.get("haptics_crowd", True) is not False
        if event.severity.value <= Severity.NAVIGATION.value:
            return True
        return False

    def _commit(self, decision: Decision, now: float) -> Decision:
        self.state.announced_at[decision.event.key] = now
        self.state.last_spoke_at = now
        self.state.spoken += 1
        self._record(decision)
        return decision

    def _reject(
        self, event: WorldEvent, reason: str, now: float,
        score: float = 0.0, threshold: float = 0.0,
    ) -> Decision:
        self.state.suppressed += 1
        self.state.suppression_reasons[reason] = (
            self.state.suppression_reasons.get(reason, 0) + 1
        )
        decision = Decision(False, False, score, threshold, reason, event)
        self._record(decision)
        return decision

    def _record(self, decision: Decision) -> None:
        self.recent_decisions.append(decision)
        del self.recent_decisions[:-40]

    # ------------------------------------------------------------------

    def mark_spoken(self, key: str) -> None:
        self.state.announced_at[key] = self._clock()

    def forget(self, key: str) -> None:
        """Allow a key to be announced again — e.g. a friend genuinely left and returned."""
        self.state.announced_at.pop(key, None)

    def stats(self) -> Dict[str, Any]:
        behaviour = self.personality.behaviour()
        return {
            "spoken": self.state.spoken,
            "suppressed": self.state.suppressed,
            "suppression_reasons": dict(self.state.suppression_reasons),
            "quiet_mode": self.quiet_mode,
            "interruption_threshold": round(behaviour.interruption_threshold, 2),
            "proactive_cooldown_s": round(behaviour.proactive_cooldown_s, 1),
        }

    def recent(self, limit: int = 12) -> List[Dict[str, Any]]:
        return [d.payload() for d in self.recent_decisions[-limit:]]
