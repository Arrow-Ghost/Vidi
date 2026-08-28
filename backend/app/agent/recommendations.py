"""
Place detection and personal recommendations — Parts 9, 34, 35, 45.

WHAT THIS CAN HONESTLY CLAIM

Part 9 forbids hallucinating that a place is beautiful, safe, empty or
enjoyable. It also asks for evidence-based detection. So every place here
carries the EVIDENCE that produced it, and the agent is told to attribute it:
"there's a sign for a library" is sayable; "there's a lovely quiet library" is
not.

A NOTE ON DISTANCE, which the spec's examples gloss over: this system has no
GPS and no maps provider, so it cannot say "50 metres ahead". Monocular depth
gives coarse ranges only. Places are therefore reported with a relative
direction and, at most, a coarse range — never a figure in metres. Saying "50
metres" would be inventing a measurement, which is the one thing this project
refuses to do.

TWO EVIDENCE SOURCES

  signage   an OCR'd sign naming a place. Strong: the place named itself.
  scene     object cues (benches, plants, greenery). Weak, and only ever
            phrased as "this looks like it might be...".

RECOMMENDATIONS ARE GATED (Part 35). Each has relevance, confidence, a
cooldown, and a priority, and a place mentioned once is not mentioned again
until the situation genuinely changes.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.profile.personality import PersonalityEngine, Trait

logger = logging.getLogger("vidi.agent.recommendations")

# Place words that can appear on signage, mapped to a category and the
# preference key that makes them personally relevant.
PLACE_KEYWORDS: Dict[str, tuple] = {
    "park": ("park", "likes_nature"),
    "garden": ("garden", "likes_nature"),
    "gardens": ("garden", "likes_nature"),
    "botanical": ("garden", "likes_nature"),
    "museum": ("museum", "likes_culture"),
    "gallery": ("gallery", "likes_culture"),
    "library": ("library", "likes_quiet"),
    "cafe": ("cafe", "likes_cafes"),
    "café": ("cafe", "likes_cafes"),
    "coffee": ("cafe", "likes_cafes"),
    "canteen": ("canteen", "likes_food"),
    "cafeteria": ("canteen", "likes_food"),
    "restaurant": ("restaurant", "likes_food"),
    "market": ("market", "likes_exploring"),
    "viewpoint": ("viewpoint", "likes_nature"),
    "trail": ("walking_path", "likes_nature"),
    "playground": ("playground", None),
    "plaza": ("plaza", None),
    "square": ("plaza", None),
    "bookshop": ("bookshop", "likes_quiet"),
    "bookstore": ("bookshop", "likes_quiet"),
}

# Detector classes that hint at an outdoor sitting area. Weak evidence only.
GREEN_SPACE_CUES = {"bench", "potted plant"}
GREEN_SPACE_MIN_CUES = 2

# Part 35: how long before the same place may be mentioned again.
PLACE_COOLDOWN_S = 300.0


@dataclass
class Place:
    category: str
    label: str
    evidence: str            # "signage" | "scene"
    confidence: float
    direction: Optional[str] = None
    direction_phrase: Optional[str] = None
    detail: str = ""

    @property
    def key(self) -> str:
        return "place:" + self.category

    def payload(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "label": self.label,
            "evidence": self.evidence,
            "confidence": round(self.confidence, 2),
            "direction": self.direction,
            "direction_phrase": self.direction_phrase,
            "detail": self.detail,
        }


@dataclass
class Recommendation:
    place: Place
    relevance: float
    reason: str
    priority: str = "low"

    def payload(self) -> Dict[str, Any]:
        return {
            **self.place.payload(),
            "relevance": round(self.relevance, 2),
            "reason": self.reason,
            "priority": self.priority,
        }


class PlaceDetector:
    """
    Finds places from evidence already gathered by other subsystems. It runs no
    model of its own — signage comes from the OCR tracker, cues from the world
    model — so it adds no perception cost.
    """

    def from_signs(self, signs: Sequence) -> List[Place]:
        found: Dict[str, Place] = {}
        for sign in signs:
            text = getattr(sign, "text", "") or ""
            for word in re.findall(r"[a-zA-Zé]+", text.lower()):
                entry = PLACE_KEYWORDS.get(word)
                if entry is None:
                    continue
                category, _preference = entry
                direction = getattr(sign, "direction", None)
                direction_value = getattr(direction, "value", None)
                confidence = min(0.9, float(getattr(sign, "confidence", 0.8)))
                place = Place(
                    category=category,
                    label=text.strip(),
                    evidence="signage",
                    confidence=confidence,
                    direction=direction_value,
                    direction_phrase=_phrase(direction_value),
                    detail="a sign reading " + repr(text.strip()),
                )
                # Signage beats scene cues, and a stronger read beats a weaker.
                if category not in found or found[category].confidence < confidence:
                    found[category] = place
        return list(found.values())

    def from_scene(self, entities: Sequence) -> List[Place]:
        """
        Weak inference from object cues. Deliberately capped at low confidence
        and phrased as a possibility, never as a fact.
        """
        cues = [e for e in entities if getattr(e, "type", "") in GREEN_SPACE_CUES]
        if len(cues) < GREEN_SPACE_MIN_CUES:
            return []
        names = sorted({e.type for e in cues})
        return [Place(
            category="green_space",
            label="an outdoor seating area",
            evidence="scene",
            confidence=0.45,
            detail="I can see " + " and ".join(names) + ", which often means a green or seating area",
        )]

    def detect(self, signs: Sequence, entities: Sequence) -> List[Place]:
        places = self.from_signs(signs)
        if not any(p.category in {"park", "garden"} for p in places):
            places.extend(self.from_scene(entities))
        return places


def _phrase(direction: Optional[str]) -> Optional[str]:
    if not direction:
        return None
    from app.agent.tools import DIRECTION_PHRASE

    return DIRECTION_PHRASE.get(direction, direction)


class RecommendationEngine:
    """
    Scores detected places against the user's stated preferences (Part 34) and
    enforces the cooldowns that stop it becoming annoying (Part 35).
    """

    # Which traits make a category worth mentioning at all.
    CATEGORY_TRAITS: Dict[str, Dict[str, float]] = {
        "park": {Trait.NATURE_LOVER.value: 0.4, Trait.EXPLORER.value: 0.2,
                 Trait.CALM.value: 0.15},
        "garden": {Trait.NATURE_LOVER.value: 0.4, Trait.CALM.value: 0.2},
        "green_space": {Trait.NATURE_LOVER.value: 0.35, Trait.CALM.value: 0.15},
        "viewpoint": {Trait.EXPLORER.value: 0.35, Trait.ADVENTURER.value: 0.3},
        "walking_path": {Trait.NATURE_LOVER.value: 0.3, Trait.EXPLORER.value: 0.25},
        "museum": {Trait.CURIOUS.value: 0.4, Trait.EXPLORER.value: 0.3},
        "gallery": {Trait.CURIOUS.value: 0.35, Trait.EXPLORER.value: 0.25},
        "library": {Trait.CURIOUS.value: 0.3, Trait.INTROVERT.value: 0.25,
                    Trait.CALM.value: 0.25},
        "bookshop": {Trait.CURIOUS.value: 0.3, Trait.INTROVERT.value: 0.2},
        "cafe": {Trait.SOCIAL.value: 0.3, Trait.EXTROVERT.value: 0.2,
                 Trait.CALM.value: 0.15},
        "canteen": {Trait.PRACTICAL.value: 0.2},
        "restaurant": {Trait.SOCIAL.value: 0.25},
        "market": {Trait.EXPLORER.value: 0.35, Trait.SPONTANEOUS.value: 0.25,
                   Trait.INTROVERT.value: -0.2},
        "plaza": {Trait.EXPLORER.value: 0.2},
        "playground": {},
    }

    def __init__(
        self,
        personality: Optional[PersonalityEngine] = None,
        clock=time.time,
        cooldown_s: float = PLACE_COOLDOWN_S,
    ) -> None:
        self.personality = personality or PersonalityEngine()
        self.detector = PlaceDetector()
        self._clock = clock
        self.cooldown_s = cooldown_s
        self._mentioned: Dict[str, float] = {}

    def relevance(self, place: Place) -> tuple:
        """Returns (score, reason) — the reason is what the agent may say."""
        profile = self.personality.profile
        vector = profile.personality
        score = 0.25 * place.confidence
        reasons: List[str] = []

        for trait, weight in self.CATEGORY_TRAITS.get(place.category, {}).items():
            contribution = (vector.get(trait) - 0.5) * 2.0 * weight
            score += contribution
            if contribution > 0.12:
                reasons.append(trait.replace("_", " "))

        # Explicit preferences carry more weight than inferred trait affinity.
        preference_key = None
        for _word, (category, key) in PLACE_KEYWORDS.items():
            if category == place.category and key:
                preference_key = key
                break
        if preference_key and profile.preferences.get(preference_key):
            score += 0.3
            reasons.append("they said they like " + preference_key[6:].replace("_", " "))

        # Signage is real evidence; scene cues are a guess.
        if place.evidence == "signage":
            score += 0.1
        else:
            score -= 0.1

        reason = ", ".join(reasons[:2]) if reasons else "it is nearby"
        return max(0.0, min(1.0, score)), reason

    def recommend(
        self, signs: Sequence, entities: Sequence, limit: int = 1,
        proactive: bool = True,
    ) -> List[Recommendation]:
        """
        Places worth mentioning, after cooldown and gating.

         distinguishes two genuinely different situations, which an
        earlier version wrongly conflated:

          proactive=True   we are considering volunteering this unasked, so a
                           user who does not want suggestions gets none.
          proactive=False  the user ASKED. Their preference not to be
                           interrupted must never become a refusal to answer a
                           direct question.
        """
        behaviour = self.personality.behaviour()
        if proactive and not behaviour.suggest_exploration:
            return []

        now = self._clock()
        results: List[Recommendation] = []
        for place in self.detector.detect(signs, entities):
            last = self._mentioned.get(place.key)
            if last is not None and now - last < self.cooldown_s:
                continue
            score, reason = self.relevance(place)
            if score < 0.35:
                continue
            priority = "medium" if score >= 0.6 else "low"
            results.append(Recommendation(place, score, reason, priority))

        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[:limit]

    def mark_mentioned(self, place: Place) -> None:
        self._mentioned[place.key] = self._clock()

    def describe(self, recommendation: Recommendation) -> str:
        """
        Phrasing that stays inside the evidence, and adapts to who is listening
        (Part 9). Never asserts a place is nice, quiet, or safe.
        """
        place = recommendation.place
        vector = self.personality.profile.personality
        where = place.direction_phrase or "nearby"

        if place.evidence == "signage":
            opening = "There's " + place.detail + " " + where
        else:
            opening = "This looks like it might be " + place.label + " — " + place.detail

        # Phrasing still follows the profile even when they asked directly.
        if not self.personality.behaviour().suggest_exploration:
            return opening + "."

        if vector.get(Trait.EXPLORER.value) >= 0.65:
            # NOT "we haven't been there before" — there is no location history
            # in this system, so that would be an unverifiable claim. The
            # invitation is personalised; the fact is not invented.
            return opening + ". Want to go and take a look?"
        if vector.get(Trait.NATURE_LOVER.value) >= 0.65 and place.category in {
            "park", "garden", "green_space", "walking_path"
        }:
            return opening + ", if you'd like somewhere green to spend a while."
        if vector.get(Trait.INTROVERT.value) >= 0.65:
            return opening + ", if you'd like somewhere to sit for a bit."
        return opening + "."

    def stats(self) -> Dict[str, Any]:
        return {
            "mentioned": len(self._mentioned),
            "cooldown_s": self.cooldown_s,
            "suggestions_enabled": self.personality.behaviour().suggest_exploration,
        }
