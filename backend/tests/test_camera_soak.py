"""
Segment 2 acceptance test: sustained frame load must not grow memory.

Two complementary claims, tested separately because they have very different
costs:

  1. The bounded queue itself never accumulates, at high volume (5000 frames,
     no inference — pure and fast).
  2. The REAL end-to-end path (decode -> detect -> track -> world model) does
     not leak across hundreds of genuine frames, and reports honest latency.

The brief asks for a 30-minute camera session. That is a manual browser check
(documented in DEVELOPMENT.md); what is automated here is the same claim in
accelerated form — memory must be flat in the number of frames, not merely
"small once".
"""

from __future__ import annotations

import asyncio
import base64
import os

import cv2
import numpy as np
import psutil
import pytest

from app.perception.pipeline import PerceptionPipeline, decode_frame, FrameDecodeError
from app.realtime.frame_queue import LatestFrameQueue


def _rss_kb() -> int:
    """
    Resident set size of this process, in KB.

    Uses psutil so this works on Windows/macOS/Linux alike. The original
    version read /proc/self/status, which made this test silently Linux-only
    — it failed the moment it was actually run on this (Windows) machine.
    psutil is already a hard dependency of ultralytics, so this costs no new
    install.
    """
    return psutil.Process(os.getpid()).memory_info().rss // 1024


def _real_jpeg_frame(width: int = 640, height: int = 480) -> str:
    """A genuine JPEG data URL — exercises the real decode path, not a stub."""
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()


def test_queue_never_accumulates_under_heavy_flood() -> None:
    """5000 frames, no consumer at all — the worst case."""

    async def run() -> None:
        queue: LatestFrameQueue[int] = LatestFrameQueue(maxsize=2)
        for i in range(5000):
            await queue.put(i)
            assert queue.qsize() <= 2
        assert queue.stats.received == 5000
        assert queue.stats.dropped == 5000 - 2
        # Latest-frame-wins: what survives must be the NEWEST frames, because
        # a stale frame is worthless to a user who has since moved.
        assert await queue.get() == 4998

    asyncio.run(run())


def test_real_perception_path_does_not_leak_across_many_frames() -> None:
    """
    Hundreds of genuine frames through the full pipeline. Memory must not
    grow proportionally to frame count.
    """
    n_frames = 250
    payload = _real_jpeg_frame()

    async def run() -> dict:
        pipeline = PerceptionPipeline()
        # Warm up first: model compilation and lazy allocations are one-time
        # costs and would otherwise be misread as a leak.
        for _ in range(5):
            await pipeline.process_payload(payload, seq=-1)

        before = _rss_kb()
        for seq in range(n_frames):
            await pipeline.process_payload(payload, seq=seq)
        after = _rss_kb()

        return {
            "growth_kb": after - before,
            "stats": pipeline.stats.snapshot(),
            "tracks": len(pipeline.tracker.tracks),
        }

    result = asyncio.run(run())

    # Frames are ~28KB each; retaining them all would be ~7MB of payload alone,
    # far more with decoded 640x480x3 arrays (~920KB each = 230MB). A flat
    # ceiling well under that proves nothing is being retained per frame.
    assert result["growth_kb"] < 60_000, (
        "RSS grew " + str(result["growth_kb"]) + "KB over " + str(n_frames) + " frames"
    )
    assert result["stats"]["frames_processed"] == n_frames + 5
    assert result["stats"]["p50_ms"] > 0
    # Random noise contains no real objects, so tracks must not pile up either.
    assert result["tracks"] < 50

    print(
        "\n[Segment 2] " + str(n_frames) + " real frames: "
        "RSS growth=" + str(result["growth_kb"]) + "KB, "
        "p50=" + str(result["stats"]["p50_ms"]) + "ms, "
        "effective_fps=" + str(result["stats"]["effective_fps"])
    )


def test_malformed_frames_are_rejected_and_counted_not_crashed() -> None:
    """Untrusted WebSocket input must never take the pipeline down."""
    for bad in ["", "not-base64!!!", "data:image/jpeg;base64,####",
                "data:image/jpeg;base64," + base64.b64encode(b"not a jpeg").decode()]:
        with pytest.raises(FrameDecodeError):
            decode_frame(bad)

    async def run() -> int:
        pipeline = PerceptionPipeline()
        for _ in range(5):
            with pytest.raises(FrameDecodeError):
                await pipeline.process_payload("garbage")
        # Still fully functional afterwards.
        result = await pipeline.process_payload(_real_jpeg_frame())
        assert result.latency_ms > 0
        return pipeline.stats.frames_failed

    assert asyncio.run(run()) == 5
