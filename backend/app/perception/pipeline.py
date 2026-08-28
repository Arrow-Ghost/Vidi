"""
PerceptionPipeline — the single consumer of the Segment 2 frame queue.

Owns the detector, tracker, and world model, and turns raw JPEG frames into
world-model updates. This is the only place the perception subsystems are
wired together, so the agent layer never touches a detector directly — it
asks the world model what is currently visible.

CONCURRENCY: inference runs in a thread executor. Even at ~15ms on the iGPU
it is blocking work, and blocking the event loop would stall the WebSocket
that carries both camera frames AND the live voice stream — audio glitches
are far more noticeable to a blind user than a dropped video frame. Frames
that arrive while a previous frame is still being processed are dropped by
the LatestFrameQueue, never queued.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import cv2
import numpy as np

from app.perception.crowd import CrowdDetectionEngine, CrowdState
from app.perception.depth import DepthEstimator, DepthMap
from app.perception.friend_tracker import FriendTracker
from app.perception.ocr import SignTracker, TextReader, TextRegion, TrackedSign
from app.perception.object_detector import Detection, ObjectDetector
from app.perception.tracker import ObjectTracker, TrackEventRecord
from app.perception.world_model import Entity, WorldModel

logger = logging.getLogger("sfcc.perception.pipeline")


class FrameDecodeError(ValueError):
    """Raised for malformed frame payloads — never silently ignored."""


def decode_frame(payload: str) -> np.ndarray:
    """
    Decode a browser 'data:image/jpeg;base64,...' payload into a BGR array.

    Validates rather than trusting: a malformed or hostile payload from a
    WebSocket client must raise, not crash the pipeline or produce a
    half-decoded buffer (see SECURITY.md — all WebSocket input is untrusted).
    """
    if not isinstance(payload, str) or not payload:
        raise FrameDecodeError("frame payload must be a non-empty string")
    encoded = payload.split(",", 1)[1] if payload.startswith("data:") else payload
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FrameDecodeError(f"invalid base64 frame payload: {exc}") from exc
    if not raw:
        raise FrameDecodeError("empty frame payload")
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise FrameDecodeError("payload was not a decodable image")
    return frame


@dataclass
class PerceptionResult:
    seq: Optional[int]
    detections: List[Detection]
    events: List[TrackEventRecord]
    entities: List[Entity]
    latency_ms: float
    image_width: int
    image_height: int
    depth_ms: Optional[float] = None
    depth_age_s: Optional[float] = None
    crowd: Optional[CrowdState] = None
    crowd_events: List[tuple] = field(default_factory=list)
    friend_events: List[tuple] = field(default_factory=list)
    faces_seen: int = 0
    face_ms: Optional[float] = None


@dataclass
class PipelineStats:
    """Honest, measured runtime numbers — surfaced by /health, never guessed."""

    frames_processed: int = 0
    frames_failed: int = 0
    _latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=120))

    def record(self, latency_ms: float) -> None:
        self.frames_processed += 1
        self._latencies.append(latency_ms)

    def snapshot(self) -> dict:
        if not self._latencies:
            return {
                "frames_processed": self.frames_processed,
                "frames_failed": self.frames_failed,
                "p50_ms": None,
                "p95_ms": None,
                "effective_fps": None,
            }
        ordered = sorted(self._latencies)
        p50 = ordered[len(ordered) // 2]
        return {
            "frames_processed": self.frames_processed,
            "frames_failed": self.frames_failed,
            "p50_ms": round(p50, 1),
            "p95_ms": round(ordered[int(len(ordered) * 0.95) - 1], 1),
            "effective_fps": round(1000 / p50, 1) if p50 > 0 else None,
        }


class PerceptionPipeline:
    # Depth costs ~47ms against detection's ~15ms and changes far more slowly
    # than object positions do, so it runs on every Nth frame and the most
    # recent map is reused in between. FusedDistanceEstimator rejects maps
    # older than its freshness window, so a stale map degrades to the size
    # prior rather than producing confidently wrong distances.
    DEPTH_EVERY_N_FRAMES = 3
    # Face recognition costs ~15-25ms per frame with faces present. At the
    # browser's 6 FPS this gives roughly 2 identity checks per second, which
    # is comfortably enough to reach 3-frame confirmation in ~1.5s.
    FACES_EVERY_N_FRAMES = 2

    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        tracker: Optional[ObjectTracker] = None,
        world_model: Optional[WorldModel] = None,
        depth: Optional[DepthEstimator] = None,
        enable_depth: bool = True,
        enable_ocr: bool = True,
        recognition: Optional[object] = None,
    ) -> None:
        self.detector = detector or ObjectDetector()
        self.tracker = tracker or ObjectTracker()
        self.world = world_model or WorldModel()
        self.depth = depth if depth is not None else (DepthEstimator() if enable_depth else None)
        self._depth_map: Optional[DepthMap] = None
        self._frame_counter = 0
        # OCR is ~25x slower than detection, so it runs on its own cadence
        # against the latest frame rather than inside the detection loop.
        # Crowd analysis is nearly free: it reads the person tracks the
        # detector already produced, so it runs every frame.
        self.crowd_engine = CrowdDetectionEngine()
        self.friend_tracker = FriendTracker()
        # Face recognition is only triggered when people are actually in
        # view, and then only every Nth frame (Part 60) — running it on
        # empty corridors would burn the budget for nothing.
        self.recognition = recognition
        self.text_reader = TextReader() if enable_ocr else None
        self.sign_tracker = SignTracker()
        self._last_frame: Optional[np.ndarray] = None
        self._ocr_lock = asyncio.Lock()
        self.last_ocr_ms: Optional[float] = None
        self.stats = PipelineStats()
        # Serializes inference: one frame in flight at a time. Without this,
        # concurrent frames would contend for the same iGPU context and
        # inflate latency for everyone.
        self._lock = asyncio.Lock()

    @property
    def backend_name(self) -> str:
        return self.detector.backend.name

    @property
    def depth_available(self) -> bool:
        return self.depth is not None and self.depth.available

    def _process_sync(self, frame: np.ndarray) -> tuple:
        detections = self.detector.detect(frame)
        height, width = frame.shape[:2]

        self._frame_counter += 1
        depth_ms = None
        if self.depth_available and self._frame_counter % self.DEPTH_EVERY_N_FRAMES == 1:
            fresh = self.depth.infer(frame)
            if fresh is not None:
                self._depth_map = fresh
                depth_ms = fresh.inference_ms

        events = self.world.apply_frame(
            self.tracker, detections, width, height, self._depth_map
        )
        depth_age = self._depth_map.age_s() if self._depth_map else None

        # ---- crowd (Parts 23-32) ----
        person_tracks = [t for t in self.tracker.tracks.values() if t.class_name == "person"]
        crowd_state, crowd_events = self.crowd_engine.update(person_tracks, width, height)

        # ---- enrolled friends (Parts 19, 58) ----
        friend_events: List[tuple] = []
        faces_seen = 0
        face_ms = None
        should_recognize = (
            self.recognition is not None
            and person_tracks
            and self._frame_counter % self.FACES_EVERY_N_FRAMES == 1
        )
        if should_recognize:
            started = time.perf_counter()
            try:
                identified, faces_seen = self.recognition.identify_people(frame)
                face_ms = (time.perf_counter() - started) * 1000
                friend_events = self.friend_tracker.update(
                    identified, width, height,
                    person_tracks=person_tracks,
                    in_crowd=crowd_state.is_crowd,
                )
            except Exception:  # noqa: BLE001 - recognition must never stop perception
                logger.exception("friend recognition failed")

        return (
            detections, events, width, height, depth_ms, depth_age,
            crowd_state, crowd_events, friend_events, faces_seen, face_ms,
        )

    async def process(self, frame: np.ndarray, seq: Optional[int] = None) -> PerceptionResult:
        started = time.perf_counter()
        # Retained (not copied) for the OCR worker; frames are read-only
        # downstream and one reference costs ~900KB, not a growing buffer.
        self._last_frame = frame
        async with self._lock:
            loop = asyncio.get_running_loop()
            # run_in_executor keeps the event loop free for voice + WebSocket I/O.
            (
                detections, events, width, height, depth_ms, depth_age,
                crowd_state, crowd_events, friend_events, faces_seen, face_ms,
            ) = await loop.run_in_executor(None, self._process_sync, frame)
        latency_ms = (time.perf_counter() - started) * 1000
        self.stats.record(latency_ms)
        return PerceptionResult(
            seq=seq,
            detections=detections,
            events=events,
            entities=list(self.world.entities.values()),
            latency_ms=round(latency_ms, 1),
            image_width=width,
            image_height=height,
            depth_ms=round(depth_ms, 1) if depth_ms else None,
            depth_age_s=round(depth_age, 2) if depth_age is not None else None,
            crowd=crowd_state,
            crowd_events=crowd_events,
            friend_events=friend_events,
            faces_seen=faces_seen,
            face_ms=round(face_ms, 1) if face_ms else None,
        )

    async def process_payload(self, payload: str, seq: Optional[int] = None) -> PerceptionResult:
        try:
            frame = decode_frame(payload)
        except FrameDecodeError:
            self.stats.frames_failed += 1
            raise
        return await self.process(frame, seq)


    # ---------------- text / signboards (Segment 15) ----------------

    @property
    def ocr_available(self) -> bool:
        return self.text_reader is not None

    async def scan_text(self) -> tuple:
        """
        Run one local OCR pass over the most recent frame.

        Returns (all_regions, newly_seen_signs). Runs under its own lock so a
        slow pass can never overlap itself, and in an executor so the ~934ms
        cost never stalls the event loop carrying the voice stream.
        """
        if self.text_reader is None or self._last_frame is None:
            return [], []
        if self._ocr_lock.locked():
            # A pass is already running; skipping is correct — the next tick
            # will read a fresher frame than this one anyway.
            return [], []

        async with self._ocr_lock:
            frame = self._last_frame
            loop = asyncio.get_running_loop()
            regions = await loop.run_in_executor(None, self.text_reader.read, frame)
            self.last_ocr_ms = self.text_reader.last_inference_ms
            new_signs = self.sign_tracker.update(regions)
            return regions, new_signs

    def current_frame(self) -> Optional[np.ndarray]:
        """Latest frame, for on-demand deep reading by the vision LLM."""
        return self._last_frame
