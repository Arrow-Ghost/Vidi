"""
Haptic feedback — Parts 38 to 44.

WHAT THE HARDWARE HERE CAN ACTUALLY DO, stated up front because Part 40
forbids claiming directional haptics a device cannot deliver:

  * This laptop has no rumble-capable controller and desktop browsers expose
    no vibration API, so the default provider is a MOCK that drives the
    dashboard. Every decision is real and visible; only the buzz is absent.
  * Opened on Android Chrome, navigator.vibrate() works. That is ONE motor,
    so it can express urgency through rhythm but CANNOT express left vs right.
    The patterns below therefore encode direction as rhythm, and
    `supports_direction` is False for that provider so nothing claims otherwise.
  * A dual-motor gamepad could express side. The abstraction allows it; we do
    not pretend to have one.

The backend owns the DECISION (which pattern, at what priority, subject to
cooldown); the browser owns the DELIVERY. That split means the same haptic
logic is testable here without any hardware attached.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("vidi.haptics")


class HapticEvent(str, Enum):
    """Part 39's event vocabulary."""

    OBSTACLE_NEAR = "OBSTACLE_NEAR"
    OBSTACLE_CRITICAL = "OBSTACLE_CRITICAL"
    PERSON_NEAR = "PERSON_NEAR"
    FRIEND_DETECTED = "FRIEND_DETECTED"
    FRIEND_APPROACHING = "FRIEND_APPROACHING"
    CROWD_DETECTED = "CROWD_DETECTED"
    CROWD_APPROACHING = "CROWD_APPROACHING"
    NAVIGATION_LEFT = "NAVIGATION_LEFT"
    NAVIGATION_RIGHT = "NAVIGATION_RIGHT"
    NAVIGATION_FORWARD = "NAVIGATION_FORWARD"
    NAVIGATION_ARRIVED = "NAVIGATION_ARRIVED"
    TASK_REMINDER = "TASK_REMINDER"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class HapticPriority(int, Enum):
    CRITICAL = 0   # never suppressed, never rate-limited away
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass(frozen=True)
class HapticPattern:
    """
    A vibration pattern as alternating on/off milliseconds — the same shape
    navigator.vibrate() takes, so the browser provider passes it straight
    through with no translation.
    """

    name: str
    pulses: List[int]
    priority: HapticPriority
    # True only for events whose MEANING is a side. Providers that cannot
    # express side must still not imply one, so this is checked at dispatch.
    directional: bool = False

    @property
    def duration_ms(self) -> int:
        return sum(self.pulses)

    def payload(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pulses": list(self.pulses),
            "priority": self.priority.name,
            "directional": self.directional,
            "duration_ms": self.duration_ms,
        }


# Part 40's tactile language, kept deliberately consistent:
#   one short pulse      informational
#   double pulse         important
#   rapid burst          urgent
#   long pulse           navigation / attention
#   long-short           left
#   short-long           right
PATTERNS: Dict[HapticEvent, HapticPattern] = {
    HapticEvent.OBSTACLE_CRITICAL: HapticPattern(
        "urgent_burst", [120, 60, 120, 60, 120, 60, 200], HapticPriority.CRITICAL),
    HapticEvent.OBSTACLE_NEAR: HapticPattern(
        "double_pulse", [100, 80, 100], HapticPriority.HIGH),
    HapticEvent.PERSON_NEAR: HapticPattern(
        "soft_double", [60, 70, 60], HapticPriority.NORMAL),
    HapticEvent.FRIEND_DETECTED: HapticPattern(
        "friend_triple", [70, 60, 70, 60, 140], HapticPriority.NORMAL),
    HapticEvent.FRIEND_APPROACHING: HapticPattern(
        "friend_rising", [50, 50, 80, 50, 130], HapticPriority.NORMAL),
    HapticEvent.CROWD_DETECTED: HapticPattern(
        "crowd_rumble", [180, 90, 180], HapticPriority.NORMAL),
    HapticEvent.CROWD_APPROACHING: HapticPattern(
        "crowd_closing", [150, 70, 150, 70, 150], HapticPriority.HIGH),
    HapticEvent.NAVIGATION_LEFT: HapticPattern(
        "long_short", [260, 90, 80], HapticPriority.HIGH, directional=True),
    HapticEvent.NAVIGATION_RIGHT: HapticPattern(
        "short_long", [80, 90, 260], HapticPriority.HIGH, directional=True),
    HapticEvent.NAVIGATION_FORWARD: HapticPattern(
        "long_pulse", [300], HapticPriority.HIGH),
    HapticEvent.NAVIGATION_ARRIVED: HapticPattern(
        "arrival", [90, 60, 90, 60, 300], HapticPriority.HIGH),
    HapticEvent.TASK_REMINDER: HapticPattern(
        "single_pulse", [110], HapticPriority.LOW),
    HapticEvent.LOW_CONFIDENCE: HapticPattern(
        "uncertain", [40, 120, 40], HapticPriority.LOW),
    HapticEvent.SYSTEM_ERROR: HapticPattern(
        "error", [220, 120, 220], HapticPriority.NORMAL),
}

# Part 42: per-event minimum gap. An obstacle that stays in view must NOT
# vibrate on every frame — only when the situation meaningfully changes.
COOLDOWN_S: Dict[HapticPriority, float] = {
    HapticPriority.CRITICAL: 2.0,
    HapticPriority.HIGH: 5.0,
    HapticPriority.NORMAL: 12.0,
    HapticPriority.LOW: 30.0,
}

# Which user preference switches each category off (Part 43).
CATEGORY_PREFERENCE = {
    HapticEvent.FRIEND_DETECTED: "haptics_friend",
    HapticEvent.FRIEND_APPROACHING: "haptics_friend",
    HapticEvent.CROWD_DETECTED: "haptics_crowd",
    HapticEvent.CROWD_APPROACHING: "haptics_crowd",
    HapticEvent.TASK_REMINDER: "haptics_task",
    HapticEvent.LOW_CONFIDENCE: "haptics_confidence",
}

SAFETY_EVENTS = {HapticEvent.OBSTACLE_CRITICAL, HapticEvent.OBSTACLE_NEAR}


@dataclass
class HapticDispatch:
    event: HapticEvent
    pattern: HapticPattern
    at: float
    intensity: float = 1.0
    note: str = ""

    def payload(self) -> Dict[str, Any]:
        return {
            "event": self.event.value,
            "intensity": round(self.intensity, 2),
            "note": self.note,
            **self.pattern.payload(),
        }


@dataclass
class HapticSettings:
    """Part 43's user-configurable surface."""

    enabled: bool = True
    intensity: float = 1.0          # scales pulse lengths
    safety_only: bool = False       # "only vibrate for important things"
    supports_direction: bool = False


class HapticEngine:
    """
    Decides whether an event should produce a buzz, which pattern, and whether
    the connected provider can honestly express it.

    Delivery happens in the browser; this returns the dispatch to send.
    """

    def __init__(
        self,
        settings: Optional[HapticSettings] = None,
        preferences: Optional[Dict[str, Any]] = None,
        clock=time.time,
    ) -> None:
        self.settings = settings or HapticSettings()
        self.preferences = preferences if preferences is not None else {}
        self._clock = clock
        self._last_fired: Dict[str, float] = {}
        self.history: List[HapticDispatch] = []
        self.provider_name = "mock"
        self.provider_connected = False
        self.suppressed = 0

    # ------------------------------------------------------------------

    def set_provider(self, name: str, connected: bool, supports_direction: bool = False) -> None:
        self.provider_name = name
        self.provider_connected = connected
        self.settings.supports_direction = supports_direction

    def trigger(
        self, event: HapticEvent, key: Optional[str] = None, force: bool = False
    ) -> Optional[HapticDispatch]:
        """
        Returns a dispatch to send to the browser, or None if this buzz should
        not happen. `key` distinguishes instances (e.g. friend:Ravi) so two
        different friends are not treated as one repeating event.
        """
        pattern = PATTERNS.get(event)
        if pattern is None:
            return None

        is_safety = event in SAFETY_EVENTS
        if not self.settings.enabled and not is_safety:
            return self._suppress()
        if self.settings.safety_only and not is_safety:
            return self._suppress()

        preference_key = CATEGORY_PREFERENCE.get(event)
        if preference_key and self.preferences.get(preference_key) is False and not is_safety:
            return self._suppress()

        # Part 42: cooldown per event key, not per event class.
        identity = event.value + ":" + (key or "")
        now = self._clock()
        if not force:
            last = self._last_fired.get(identity)
            if last is not None and now - last < COOLDOWN_S[pattern.priority]:
                return self._suppress()

        note = ""
        if pattern.directional and not self.settings.supports_direction:
            # Honesty: keep the rhythm, but do not let anything downstream
            # claim the buzz itself indicated a side.
            note = (
                "Provider has a single motor and cannot convey direction; this pattern "
                "signals a turn but not which way. The spoken instruction carries the side."
            )

        self._last_fired[identity] = now
        dispatch = HapticDispatch(
            event=event,
            pattern=self._scaled(pattern),
            at=now,
            intensity=self.settings.intensity,
            note=note,
        )
        self.history.append(dispatch)
        del self.history[:-30]
        return dispatch

    def _scaled(self, pattern: HapticPattern) -> HapticPattern:
        """Intensity scales pulse length — the only knob a single motor has."""
        factor = max(0.2, min(1.5, self.settings.intensity))
        if abs(factor - 1.0) < 0.01:
            return pattern
        return HapticPattern(
            name=pattern.name,
            pulses=[max(20, int(p * factor)) for p in pattern.pulses],
            priority=pattern.priority,
            directional=pattern.directional,
        )

    def _suppress(self) -> None:
        self.suppressed += 1
        return None

    # ------------------------------------------------------------------

    def apply_preferences(self, preferences: Dict[str, Any]) -> None:
        self.preferences = preferences
        if preferences.get("haptics_enabled") is False:
            self.settings.enabled = False
        elif preferences.get("haptics_enabled") is True:
            self.settings.enabled = True
        if preferences.get("haptics_safety_only") is True:
            self.settings.safety_only = True
        elif preferences.get("haptics_safety_only") is False:
            self.settings.safety_only = False

    def status(self) -> Dict[str, Any]:
        last = self.history[-1] if self.history else None
        return {
            "enabled": self.settings.enabled,
            "safety_only": self.settings.safety_only,
            "intensity": round(self.settings.intensity, 2),
            "provider": self.provider_name,
            "connected": self.provider_connected,
            "supports_direction": self.settings.supports_direction,
            "fired": len(self.history),
            "suppressed": self.suppressed,
            "last": last.payload() if last else None,
        }


# Mapping from world events to haptic events, so the two vocabularies stay
# decoupled — a new world event does not require touching the pattern table.
def haptic_for_event(event_type: str, severity_name: str) -> Optional[HapticEvent]:
    mapping = {
        "OBSTACLE_DETECTED": HapticEvent.OBSTACLE_NEAR,
        "FRIEND_DETECTED": HapticEvent.FRIEND_DETECTED,
        "FRIEND_DETECTED_IN_CROWD": HapticEvent.FRIEND_DETECTED,
        "FRIEND_APPROACHING": HapticEvent.FRIEND_APPROACHING,
        "CROWD_APPEARED": HapticEvent.CROWD_DETECTED,
        "CROWD_APPROACHING": HapticEvent.CROWD_APPROACHING,
        "CROWD_CONGESTION": HapticEvent.CROWD_DETECTED,
        "TASK_DUE": HapticEvent.TASK_REMINDER,
        "WARNING_SIGN_DETECTED": HapticEvent.OBSTACLE_NEAR,
        "SYSTEM_DEGRADED": HapticEvent.SYSTEM_ERROR,
    }
    if severity_name == "SAFETY":
        return HapticEvent.OBSTACLE_CRITICAL
    return mapping.get(event_type)
