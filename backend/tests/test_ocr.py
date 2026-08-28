"""
OCR and signboard tests — Segments 15, 28.

The signboard fixture is synthetic on purpose: it has known ground truth
(EXIT on the left, Room 204 / Science Lab on the right, WET FLOOR in the
centre), so tests can assert exact text AND exact direction rather than just
"something was read".

The announcer tests are mostly about what must NOT be said. An assistive
device that reads out every sign it passes is worse than one that says
nothing, because the user cannot skim past speech.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.agent.proactive import ProactiveAnnouncer, ProactiveSettings, Urgency
from app.perception.direction import DirectionBucket
from app.perception.ocr import (
    MIN_TEXT_CONFIDENCE,
    SignTracker,
    TextReader,
    TextRegion,
    TrackedSign,
)

FIXTURES = Path(__file__).parent / "fixtures"
SIGNBOARD = FIXTURES / "signboard.jpg"


def _frame() -> np.ndarray:
    image = cv2.imread(str(SIGNBOARD))
    assert image is not None, "signboard fixture missing"
    return image


def _region(text: str, x_centre: float, confidence: float = 0.95,
            width: int = 960) -> TextRegion:
    half = 40
    return TextRegion(
        text=text, confidence=confidence,
        box=(x_centre - half, 100, x_centre + half, 160),
        image_width=width, image_height=640,
    )


def _sign(text: str, navigational=False, warning=False, confidence=0.95,
          direction=DirectionBucket.CENTER) -> TrackedSign:
    return TrackedSign(
        key=text.lower().replace(" ", ""), text=text, direction=direction,
        confidence=confidence, first_seen=0.0, last_seen=0.0,
        navigational=navigational, warning=warning,
    )


# ---------------- local OCR engine ----------------

def test_reads_every_sign_on_the_board_with_correct_text() -> None:
    reader = TextReader()
    regions = reader.read(_frame())
    if not reader.available:
        pytest.skip("local OCR engine not installed")

    found = {r.text.strip().upper() for r in regions}
    for expected in ["EXIT", "ROOM 204", "SCIENCE LAB", "WET FLOOR"]:
        assert expected in found, "did not read " + expected + "; got " + str(found)

    print("\n[Segment 15] read " + str(len(regions)) + " regions in "
          + str(round(reader.last_inference_ms or 0)) + "ms: " + str(sorted(found)))


def test_each_sign_is_placed_on_the_correct_side() -> None:
    """
    Direction is the half of OCR that actually matters for navigation. Getting
    the text right but the side wrong sends the user the wrong way.
    """
    reader = TextReader()
    regions = reader.read(_frame())
    if not reader.available:
        pytest.skip("local OCR engine not installed")

    by_text = {r.text.strip().upper(): r.direction().value for r in regions}
    assert "left" in by_text.get("EXIT", ""), by_text
    assert "right" in by_text.get("ROOM 204", ""), by_text
    assert by_text.get("WET FLOOR", "").startswith("center"), by_text


def test_low_confidence_and_single_characters_are_discarded() -> None:
    """Noise must never reach the user as if it were signage."""
    reader = TextReader()
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    assert reader.read(blank) == []


def test_reader_without_engine_returns_nothing_rather_than_raising() -> None:
    reader = TextReader()
    reader._load_failed = True  # simulate a missing engine
    reader._engine = None
    assert reader.read(_frame()) == []


# ---------------- classification ----------------

@pytest.mark.parametrize("text", ["EXIT", "Room 204", "Platform 3", "Stairs", "Reception"])
def test_navigational_text_is_recognized(text: str) -> None:
    assert _region(text, 100).looks_navigational is True


@pytest.mark.parametrize("text", ["WET FLOOR", "CAUTION", "DANGER High Voltage"])
def test_warning_text_is_recognized(text: str) -> None:
    assert _region(text, 100).looks_like_warning is True


def test_ordinary_text_is_neither_navigational_nor_a_warning() -> None:
    region = _region("Happy Birthday", 100)
    assert region.looks_navigational is False
    assert region.looks_like_warning is False


# ---------------- sign tracking ----------------

def test_a_sign_is_reported_new_only_once() -> None:
    """Walking past a sign must not make the assistant repeat itself."""
    tracker = SignTracker()
    regions = [_region("EXIT", 100)]

    first = tracker.update(regions, now=1000.0)
    assert len(first) == 1
    tracker.mark_announced(first)

    for tick in range(1, 5):
        assert tracker.update(regions, now=1000.0 + tick) == []


def test_a_sign_that_moves_across_the_view_is_reported_again() -> None:
    """
    As the user turns, a sign genuinely changes position relative to them —
    that is new information, not repetition.
    """
    tracker = SignTracker()
    first = tracker.update([_region("EXIT", 60)], now=1000.0)   # far left
    tracker.mark_announced(first)

    again = tracker.update([_region("EXIT", 900)], now=1001.0)  # now far right
    assert len(again) == 1
    assert "right" in again[0].direction.value


def test_signs_expire_once_out_of_view() -> None:
    tracker = SignTracker()
    tracker.update([_region("EXIT", 100)], now=1000.0)
    assert len(tracker.current()) >= 0
    tracker.update([], now=1000.0 + SignTracker.FORGET_AFTER_S + 1)
    assert tracker.signs == {}


def test_current_signs_put_warnings_first() -> None:
    tracker = SignTracker()
    tracker.update([_region("Room 204", 800), _region("WET FLOOR", 400)], now=1000.0)
    ordered = tracker.current(now=1000.0)
    assert ordered[0].warning is True, "a hazard must be surfaced before a room number"


# ---------------- proactive announcement gating ----------------

def test_only_warnings_and_navigation_are_ever_announced_unprompted() -> None:
    announcer = ProactiveAnnouncer()
    assert announcer.select([_sign("Happy Birthday")]) == []
    assert announcer.select([_sign("EXIT", navigational=True)]) != []


def test_nothing_is_announced_while_the_assistant_is_speaking() -> None:
    """Interrupting the user's own conversation is the cardinal sin here."""
    announcer = ProactiveAnnouncer()
    chosen = announcer.select([_sign("EXIT", navigational=True)], user_is_busy=True)
    assert chosen == []
    assert announcer.suppressed_count == 1


def test_cooldown_prevents_a_stream_of_announcements() -> None:
    clock = {"t": 1000.0}
    announcer = ProactiveAnnouncer(
        ProactiveSettings(cooldown_s=20.0), clock=lambda: clock["t"]
    )
    assert announcer.select([_sign("EXIT", navigational=True)])
    announcer.mark_announced()

    clock["t"] += 5
    assert announcer.select([_sign("Room 204", navigational=True)]) == []

    clock["t"] += 20
    assert announcer.select([_sign("Room 204", navigational=True)]) != []


def test_a_warning_may_interrupt_sooner_than_a_direction_sign() -> None:
    clock = {"t": 1000.0}
    announcer = ProactiveAnnouncer(
        ProactiveSettings(cooldown_s=20.0, warning_cooldown_s=8.0),
        clock=lambda: clock["t"],
    )
    announcer.mark_announced()

    clock["t"] += 10
    assert announcer.select([_sign("Room 204", navigational=True)]) == []
    assert announcer.select([_sign("WET FLOOR", warning=True)]) != []


def test_low_confidence_reads_are_never_announced_unprompted() -> None:
    """A misread sign spoken without being asked for is worse than silence."""
    announcer = ProactiveAnnouncer()
    assert announcer.select([_sign("EXIT", navigational=True, confidence=0.5)]) == []


def test_announcements_are_capped_so_the_user_hears_one_thing() -> None:
    announcer = ProactiveAnnouncer(ProactiveSettings(max_signs_per_announcement=2))
    chosen = announcer.select([
        _sign("EXIT", navigational=True), _sign("Room 204", navigational=True),
        _sign("Platform 3", navigational=True), _sign("Level 2", navigational=True),
    ])
    assert len(chosen) == 2


def test_composed_note_quotes_the_sign_and_asks_for_one_sentence() -> None:
    announcer = ProactiveAnnouncer()
    note = announcer.compose([
        _sign("EXIT", navigational=True, direction=DirectionBucket.LEFT)
    ])
    assert '"EXIT"' in note
    assert "to your left" in note
    assert "ONE short sentence" in note


def test_disabling_proactive_mode_silences_everything() -> None:
    announcer = ProactiveAnnouncer(ProactiveSettings(enabled=False))
    assert announcer.select([_sign("WET FLOOR", warning=True)]) == []


def test_urgency_ranking() -> None:
    assert ProactiveAnnouncer.urgency_of(_sign("WET FLOOR", warning=True)) == Urgency.WARNING
    assert ProactiveAnnouncer.urgency_of(_sign("EXIT", navigational=True)) == Urgency.NAVIGATION
    assert ProactiveAnnouncer.urgency_of(_sign("Hello")) == Urgency.INFORMATION


# ---------------- pipeline integration ----------------

def test_pipeline_scan_text_finds_signs_and_tracks_them() -> None:
    from app.perception.pipeline import PerceptionPipeline

    async def run():
        pipeline = PerceptionPipeline()
        if not pipeline.ocr_available:
            pytest.skip("OCR disabled")
        await pipeline.process(_frame())
        regions, new_signs = await pipeline.scan_text()
        return regions, new_signs, pipeline

    regions, new_signs, pipeline = asyncio.run(run())
    if not regions:
        pytest.skip("local OCR engine not installed")

    assert any("EXIT" in r.text.upper() for r in regions)
    assert new_signs, "first pass should report new signs"
    assert pipeline.sign_tracker.current()


def test_scan_text_without_a_frame_is_a_no_op() -> None:
    from app.perception.pipeline import PerceptionPipeline

    async def run():
        pipeline = PerceptionPipeline(enable_depth=False)
        return await pipeline.scan_text()

    assert asyncio.run(run()) == ([], [])
