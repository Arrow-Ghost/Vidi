"""
Proactive announcements — Segment 28.

Lets the assistant speak without being asked — "there's a sign on your left
that says EXIT" — while making that genuinely rare.

An assistant that narrates every sign it sees is worse than one that stays
quiet, because a blind user cannot skim past it. So every candidate
announcement has to clear four independent gates:

  1. URGENCY   — warnings outrank navigation; everything else is silent.
  2. NOVELTY   — a sign is announced once. SignTracker handles re-entry.
  3. COOLDOWN  — a minimum gap between any two announcements.
  4. QUIET     — nothing while the user or the assistant is mid-turn.

Gate 4 matters most in practice: interrupting someone's question to read out
a corridor sign is exactly the behaviour that makes people switch an
assistive device off.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from app.agent.tools import DIRECTION_PHRASE
from app.perception.ocr import TrackedSign

logger = logging.getLogger("vidi.agent.proactive")


class Urgency:
    WARNING = 0      # wet floor, caution, closed
    NAVIGATION = 1   # exit, room numbers, platforms
    INFORMATION = 2  # everything else — never announced unprompted


@dataclass
class ProactiveSettings:
    enabled: bool = True
    # Minimum gap between announcements, regardless of how much appears.
    cooldown_s: float = 20.0
    # Warnings may jump the queue sooner than ordinary navigation signs.
    warning_cooldown_s: float = 8.0
    # Never announce a reading we are not confident about — a misread sign
    # spoken unprompted is worse than silence.
    min_confidence: float = 0.80
    # Cap per announcement so the user hears one useful thing, not a list.
    max_signs_per_announcement: int = 2


class ProactiveAnnouncer:
    def __init__(
        self,
        settings: Optional[ProactiveSettings] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings or ProactiveSettings()
        self._clock = clock
        self._last_announced_at: float = 0.0
        self.suppressed_count = 0
        self.announced_count = 0

    @staticmethod
    def urgency_of(sign: TrackedSign) -> int:
        if sign.warning:
            return Urgency.WARNING
        if sign.navigational:
            return Urgency.NAVIGATION
        return Urgency.INFORMATION

    def select(
        self, candidates: Sequence[TrackedSign], user_is_busy: bool = False
    ) -> List[TrackedSign]:
        """
        Decide what (if anything) is worth interrupting for. Returns the signs
        to announce; empty means stay quiet.
        """
        if not self.settings.enabled or not candidates:
            return []

        if user_is_busy:
            # Mid-conversation. Say nothing; the sign stays un-announced and
            # will be offered again on a later pass if still visible.
            self.suppressed_count += 1
            return []

        worthy = [
            s for s in candidates
            if self.urgency_of(s) != Urgency.INFORMATION
            and s.confidence >= self.settings.min_confidence
        ]
        if not worthy:
            return []

        worthy.sort(key=lambda s: (self.urgency_of(s), -s.confidence))
        has_warning = self.urgency_of(worthy[0]) == Urgency.WARNING
        required_gap = (
            self.settings.warning_cooldown_s if has_warning else self.settings.cooldown_s
        )

        now = self._clock()
        if now - self._last_announced_at < required_gap:
            self.suppressed_count += 1
            return []

        return worthy[: self.settings.max_signs_per_announcement]

    def compose(self, signs: Sequence[TrackedSign]) -> str:
        """
        Build the note handed to the model. It states the facts and asks for
        one short sentence — the model must not embellish or add signs.
        """
        parts = []
        for sign in signs:
            where = DIRECTION_PHRASE.get(sign.direction.value, sign.direction.value)
            label = "warning sign" if sign.warning else "sign"
            parts.append('a ' + label + ' ' + where + ' reading "' + sign.text + '"')
        joined = parts[0] if len(parts) == 1 else " and ".join(parts)
        return (
            "[PERCEPTION EVENT] The camera has just read " + joined + ". "
            "Tell the user in ONE short sentence. Quote the sign's wording exactly. "
            "Do not mention anything else and do not ask a follow-up question."
        )

    def mark_announced(self) -> None:
        self._last_announced_at = self._clock()
        self.announced_count += 1

    def stats(self) -> dict:
        return {
            "enabled": self.settings.enabled,
            "announced": self.announced_count,
            "suppressed": self.suppressed_count,
            "cooldown_s": self.settings.cooldown_s,
        }
