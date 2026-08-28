"""
Text and signboard reading — Segment 15.

TWO TIERS, because they solve different problems (both measured, see
docs/PERFORMANCE.md):

  Local RapidOCR (PP-OCRv6)   ~934 ms, free, offline, runs continuously
  Gemini vision              ~3500 ms, costs credits, on demand only

The local engine is what lets the assistant *notice* signs without being
asked — it runs on a slow cadence and reports text plus position, which is
enough to say "there's a sign on your left that says EXIT". Gemini is what
gets called when the user actually wants the text understood: what it means,
what it implies for navigation, or a summary of a document.

Reading text is not the same as understanding it, and this module keeps that
distinction: `TextRegion` carries only what was literally read, with a real
confidence. Meaning is added later, by the LLM, and only when asked.

NEVER HALLUCINATE TEXT: anything below the confidence floor is dropped rather
than guessed at. Telling a blind user a door says "EXIT" when it says
"EMERGENCY ONLY — ALARM WILL SOUND" is exactly the failure this must avoid.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.perception.direction import DirectionBucket, _raw_bucket

logger = logging.getLogger("sfcc.perception.ocr")

# Below this the read is treated as unreliable and discarded. PP-OCR returns
# well-calibrated scores; 0.6 keeps confident reads and drops noise/artifacts.
MIN_TEXT_CONFIDENCE = 0.60

# A single character read in isolation is almost always noise on a wall or a
# texture edge, not signage.
MIN_TEXT_LENGTH = 2

# Words that make a sign navigationally relevant. Used only as a cheap local
# hint — the authoritative judgement comes from the LLM when asked.
_NAV_HINTS = {
    "exit", "entrance", "enter", "in", "out", "way", "stairs", "stair", "lift",
    "elevator", "escalator", "toilet", "restroom", "wc", "washroom", "platform",
    "gate", "floor", "level", "room", "hall", "block", "wing", "reception",
    "office", "library", "lab", "canteen", "cafeteria", "parking", "north",
    "south", "east", "west", "left", "right", "up", "down", "emergency",
}
_WARNING_HINTS = {
    "caution", "warning", "danger", "wet", "slippery", "closed", "stop",
    "do", "not", "no", "restricted", "authorised", "authorized", "alarm",
    "construction", "hazard", "high", "voltage",
}


@dataclass
class TextRegion:
    """One piece of text actually read from the image. No interpretation."""

    text: str
    confidence: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels
    image_width: int
    image_height: int

    @property
    def center_x(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])

    def direction(self) -> DirectionBucket:
        if self.image_width <= 0:
            return DirectionBucket.CENTER
        return _raw_bucket(self.center_x / self.image_width)

    def normalized_key(self) -> str:
        """Stable identity for deduplication across frames."""
        return re.sub(r"[^a-z0-9]+", "", self.text.lower())

    @property
    def looks_navigational(self) -> bool:
        words = set(re.findall(r"[a-z]+", self.text.lower()))
        if words & _NAV_HINTS:
            return True
        # "Room 204", "Platform 3", "Gate 12" — a word plus a number.
        return bool(re.search(r"\b[a-zA-Z]+\s*\d+\b", self.text))

    @property
    def looks_like_warning(self) -> bool:
        return bool(set(re.findall(r"[a-z]+", self.text.lower())) & _WARNING_HINTS)

    def payload(self) -> Dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 2),
            "direction": self.direction().value,
            "navigational": self.looks_navigational,
            "warning": self.looks_like_warning,
        }


class TextReader:
    """
    Local OCR engine. Loads lazily — the models cost ~400 ms to initialize and
    a deployment that never reads text should not pay that at startup.
    """

    def __init__(self, min_confidence: float = MIN_TEXT_CONFIDENCE) -> None:
        self.min_confidence = min_confidence
        self.available = False
        self.last_inference_ms: Optional[float] = None
        self._engine = None
        self._load_failed = False

    def _ensure_engine(self) -> bool:
        if self._engine is not None:
            return True
        if self._load_failed:
            return False
        try:
            from rapidocr import RapidOCR

            self._engine = RapidOCR()
            self.available = True
            logger.info("local OCR engine ready (RapidOCR / PP-OCRv6)")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("local OCR unavailable: %s", exc)
            self._load_failed = True
            return False

    def read(self, frame: np.ndarray) -> List[TextRegion]:
        """
        Returns every confidently-read text region. Never raises: OCR failing
        must degrade to "I can't read that" rather than breaking perception.
        """
        if not self._ensure_engine():
            return []
        height, width = frame.shape[:2]
        try:
            started = time.perf_counter()
            result = self._engine(frame)
            self.last_inference_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR inference failed: %s", exc)
            return []

        # `x or []` cannot be used here: RapidOCR returns numpy arrays, and
        # truth-testing a multi-element array raises ValueError.
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        boxes = getattr(result, "boxes", None)
        if texts is None or scores is None or boxes is None:
            return []

        regions: List[TextRegion] = []
        for text, score, quad in zip(texts, scores, boxes):
            text = (text or "").strip()
            confidence = float(score)
            if len(text) < MIN_TEXT_LENGTH or confidence < self.min_confidence:
                continue
            xs = [float(p[0]) for p in quad]
            ys = [float(p[1]) for p in quad]
            regions.append(
                TextRegion(
                    text=text,
                    confidence=confidence,
                    box=(min(xs), min(ys), max(xs), max(ys)),
                    image_width=width,
                    image_height=height,
                )
            )
        # Largest text first: on a signboard the biggest line is the message.
        regions.sort(key=lambda r: r.height, reverse=True)
        return regions


@dataclass
class TrackedSign:
    """A sign the system has seen, remembered so it is announced only once."""

    key: str
    text: str
    direction: DirectionBucket
    confidence: float
    first_seen: float
    last_seen: float
    announced: bool = False
    navigational: bool = False
    warning: bool = False

    def payload(self) -> Dict:
        return {
            "text": self.text,
            "direction": self.direction.value,
            "confidence": round(self.confidence, 2),
            "navigational": self.navigational,
            "warning": self.warning,
            "seconds_since_seen": round(time.time() - self.last_seen, 1),
        }


class SignTracker:
    """
    Keeps track of which signs are currently visible and which are new.

    Without this the assistant would re-read the same sign on every OCR pass,
    which for a user walking down a corridor would be unbearable. A sign is
    announced once; it becomes announceable again only after it has genuinely
    left view and come back.
    """

    # How long a sign stays "current" after it was last read.
    FORGET_AFTER_S = 12.0
    # Movement of this many direction buckets counts as a real change worth
    # re-reporting (e.g. a sign moving from ahead to your left as you turn).
    DIRECTION_CHANGE_BUCKETS = 2

    _ORDER = [
        DirectionBucket.FAR_LEFT, DirectionBucket.LEFT, DirectionBucket.CENTER_LEFT,
        DirectionBucket.CENTER, DirectionBucket.CENTER_RIGHT, DirectionBucket.RIGHT,
        DirectionBucket.FAR_RIGHT,
    ]

    def __init__(self) -> None:
        self.signs: Dict[str, TrackedSign] = {}

    def update(self, regions: Sequence[TextRegion], now: Optional[float] = None) -> List[TrackedSign]:
        """Feed one OCR pass; returns the signs that are NEW (never announced)."""
        now = now if now is not None else time.time()
        new_signs: List[TrackedSign] = []

        for region in regions:
            key = region.normalized_key()
            if not key:
                continue
            existing = self.signs.get(key)
            direction = region.direction()

            if existing is None:
                sign = TrackedSign(
                    key=key, text=region.text, direction=direction,
                    confidence=region.confidence, first_seen=now, last_seen=now,
                    navigational=region.looks_navigational,
                    warning=region.looks_like_warning,
                )
                self.signs[key] = sign
                new_signs.append(sign)
                continue

            moved = abs(
                self._ORDER.index(direction) - self._ORDER.index(existing.direction)
            ) >= self.DIRECTION_CHANGE_BUCKETS
            existing.last_seen = now
            existing.confidence = max(existing.confidence, region.confidence)
            if moved:
                existing.direction = direction
                # It has genuinely moved relative to the user; worth saying again.
                existing.announced = False
                new_signs.append(existing)

        self._expire(now)
        return [s for s in new_signs if not s.announced]

    def _expire(self, now: float) -> None:
        for key in [k for k, s in self.signs.items() if now - s.last_seen > self.FORGET_AFTER_S]:
            del self.signs[key]

    def mark_announced(self, signs: Sequence[TrackedSign]) -> None:
        for sign in signs:
            sign.announced = True

    def current(self, max_age_s: float = 6.0, now: Optional[float] = None) -> List[TrackedSign]:
        now = now if now is not None else time.time()
        visible = [s for s in self.signs.values() if now - s.last_seen <= max_age_s]
        # Navigational and warning signs first — they are what a moving user
        # actually needs.
        visible.sort(
            key=lambda s: (not s.warning, not s.navigational, -s.confidence)
        )
        return visible
