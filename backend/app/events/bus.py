"""
Event bus and world events — Parts 32, 47, 48.

Perception used to talk almost directly to speech. That worked while signage
was the only proactive source, but it does not scale to friends, crowds, and
recommendations all competing for the same one voice channel.

Everything that happens in the world now becomes a typed WorldEvent on a bus.
The RelevanceEngine scores it, the PersonalityEngine shapes it, and the
AttentionManager decides — deterministically — whether it is worth
interrupting a blind user for. The LLM never makes that call, because
"should I speak right now" must be reproducible and testable (Part 49).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional

logger = logging.getLogger("sfcc.events")


class EventType(str, Enum):
    # --- objects / obstacles ---
    OBJECT_DETECTED = "OBJECT_DETECTED"
    OBSTACLE_DETECTED = "OBSTACLE_DETECTED"

    # --- people ---
    PERSON_DETECTED = "PERSON_DETECTED"
    FRIEND_DETECTED = "FRIEND_DETECTED"
    FRIEND_APPROACHING = "FRIEND_APPROACHING"
    FRIEND_MOVED = "FRIEND_MOVED"
    FRIEND_OCCLUDED = "FRIEND_OCCLUDED"
    FRIEND_LOST = "FRIEND_LOST"
    FRIEND_CONFIDENCE_DECREASED = "FRIEND_CONFIDENCE_DECREASED"
    FRIEND_DETECTED_IN_CROWD = "FRIEND_DETECTED_IN_CROWD"
    FRIEND_LOST_IN_CROWD = "FRIEND_LOST_IN_CROWD"

    # --- crowds (Part 32) ---
    CROWD_APPEARED = "CROWD_APPEARED"
    CROWD_DENSITY_INCREASED = "CROWD_DENSITY_INCREASED"
    CROWD_DENSITY_DECREASED = "CROWD_DENSITY_DECREASED"
    CROWD_STARTED_MOVING = "CROWD_STARTED_MOVING"
    CROWD_STOPPED = "CROWD_STOPPED"
    CROWD_DIRECTION_CHANGED = "CROWD_DIRECTION_CHANGED"
    CROWD_APPROACHING = "CROWD_APPROACHING"
    CROWD_DEPARTING = "CROWD_DEPARTING"
    CROWD_CONGESTION = "CROWD_CONGESTION"
    CROWD_DISPERSED = "CROWD_DISPERSED"

    # --- text / places ---
    SIGN_DETECTED = "SIGN_DETECTED"
    WARNING_SIGN_DETECTED = "WARNING_SIGN_DETECTED"
    PLACE_DETECTED = "PLACE_DETECTED"
    RECOMMENDATION = "RECOMMENDATION"

    # --- tasks / system ---
    TASK_DUE = "TASK_DUE"
    SYSTEM_DEGRADED = "SYSTEM_DEGRADED"


class Severity(int, Enum):
    """
    Fixed importance floor per event class (Part 37).

    SAFETY events are the reason this is an enum and not a tunable number:
    personality may raise or lower how chatty the assistant is, but no profile
    setting can demote something out of SAFETY (Part 56).
    """

    SAFETY = 0        # immediate hazard — always speaks
    NAVIGATION = 1    # user-requested target, route-critical
    SOCIAL = 2        # trusted friends
    ENVIRONMENT = 3   # crowd changes, task reminders
    AMBIENT = 4       # interesting places, casual commentary


@dataclass
class WorldEvent:
    type: EventType
    severity: Severity
    # Stable identity for deduplication — e.g. "friend:Ravi", "crowd:main".
    key: str
    # What a human would be told. The agent may rephrase but not embellish.
    summary: str
    confidence: float = 1.0
    data: Dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def payload(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "severity": self.severity.name,
            "key": self.key,
            "summary": self.summary,
            "confidence": round(self.confidence, 2),
            "data": self.data,
        }


Subscriber = Callable[[WorldEvent], Awaitable[None]]


class EventBus:
    """
    Small in-process pub/sub with a bounded history.

    Bounded on purpose: this runs for hours in a browser session, and an
    unbounded event log is the same memory leak as an unbounded frame queue.
    """

    def __init__(self, history: int = 200) -> None:
        self._subscribers: List[Subscriber] = []
        self.history: Deque[WorldEvent] = deque(maxlen=history)
        self.counts: Dict[str, int] = {}

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.append(subscriber)

    async def publish(self, event: WorldEvent) -> None:
        self.history.append(event)
        self.counts[event.type.value] = self.counts.get(event.type.value, 0) + 1
        for subscriber in list(self._subscribers):
            try:
                await subscriber(event)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one bad subscriber must not stop the rest
                logger.exception("event subscriber failed for %s", event.type.value)

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        return [e.payload() for e in list(self.history)[-limit:]]

    def stats(self) -> Dict[str, Any]:
        return {"total": sum(self.counts.values()), "by_type": dict(self.counts)}
