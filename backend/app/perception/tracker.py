"""
ObjectTracker — Segment 3.

Turns per-frame Detection lists into a stable set of tracked entities with
persistent IDs, so the rest of the system reasons about "the backpack that
was here a moment ago" instead of re-identifying everything from scratch
every frame. Matching is IoU-based — deliberately simple and fast enough
for CPU-only real-time use, no re-identification model involved. A class
mismatch never matches, regardless of overlap.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

from app.perception.object_detector import BoundingBox, Detection


class TrackEvent(str, Enum):
    ENTERED = "OBJECT_ENTERED"
    UPDATED = "OBJECT_UPDATED"
    MOVED = "OBJECT_MOVED"
    LEFT = "OBJECT_LEFT"
    LOST = "OBJECT_LOST"


@dataclass
class Track:
    id: str
    class_name: str
    box: BoundingBox
    confidence: float
    last_seen: float
    first_seen: float
    misses: int = 0
    hits: int = 1
    # Velocity in normalized-frame units per second. Crowd flow, approach
    # detection and "is my friend coming toward me" all need motion, and
    # deriving it here means one shared, already-smoothed source of truth
    # rather than each consumer differencing boxes itself.
    vx: float = 0.0
    vy: float = 0.0
    # Apparent size change per second: growing means approaching the camera.
    scale_rate: float = 0.0

    def update_motion(
        self, new_box: BoundingBox, dt: float,
        image_width: float, image_height: float, smoothing: float = 0.6,
    ) -> None:
        """
        Exponentially-smoothed velocity. Raw frame-to-frame deltas are far too
        jittery at 6 FPS with a hand-held camera to drive "the crowd is moving
        left", so each update blends into the previous estimate.
        """
        if dt <= 0 or image_width <= 0 or image_height <= 0:
            return
        dx = (new_box.center_x - self.box.center_x) / image_width / dt
        dy = (new_box.center_y - self.box.center_y) / image_height / dt
        previous_height = max(self.box.height, 1e-6)
        ds = (new_box.height - self.box.height) / previous_height / dt

        self.vx = smoothing * self.vx + (1 - smoothing) * dx
        self.vy = smoothing * self.vy + (1 - smoothing) * dy
        self.scale_rate = smoothing * self.scale_rate + (1 - smoothing) * ds

    @property
    def speed(self) -> float:
        return (self.vx ** 2 + self.vy ** 2) ** 0.5

    def move_distance(self, new_box: BoundingBox) -> float:
        return (
            (self.box.center_x - new_box.center_x) ** 2
            + (self.box.center_y - new_box.center_y) ** 2
        ) ** 0.5


@dataclass
class TrackEventRecord:
    event: TrackEvent
    track_id: str
    class_name: str
    box: BoundingBox
    confidence: float


def iou(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


@dataclass
class TrackerConfig:
    iou_match_threshold: float = 0.3
    max_misses_before_lost: int = 5  # consecutive unmatched frames before LEFT
    move_threshold_px: float = 20.0  # min center movement to emit MOVED vs UPDATED
    duplicate_iou_threshold: float = 0.85  # same-class overlap treated as one detection


class ObjectTracker:
    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: Dict[str, Track] = {}
        self._id_counter = itertools.count(1)

    def _next_id(self, class_name: str) -> str:
        return f"{class_name.replace(' ', '_')}_{next(self._id_counter)}"

    def _suppress_duplicates(self, detections: List[Detection]) -> List[Detection]:
        """Heavily-overlapping same-class boxes collapse to one (highest confidence wins)."""
        kept: List[Detection] = []
        used = [False] * len(detections)
        order = sorted(range(len(detections)), key=lambda i: detections[i].confidence, reverse=True)
        for i in order:
            if used[i]:
                continue
            kept.append(detections[i])
            used[i] = True
            for j in order:
                if used[j] or i == j:
                    continue
                if detections[i].class_name != detections[j].class_name:
                    continue
                if iou(detections[i].box, detections[j].box) >= self.config.duplicate_iou_threshold:
                    used[j] = True
        return kept

    def update(self, detections: List[Detection], now: Optional[float] = None) -> List[TrackEventRecord]:
        """
        Feed one frame's detections in; get back the events describing what
        changed (entered/updated/moved/lost/left). This is the first real
        consumer of Segment 2's frame queue once wired into the pipeline.
        """
        now = now if now is not None else time.time()
        detections = self._suppress_duplicates(detections)
        events: List[TrackEventRecord] = []

        unmatched_detections = list(range(len(detections)))
        matched_track_ids: Set[str] = set()

        pairs = []
        for track_id, track in self.tracks.items():
            for di in unmatched_detections:
                det = detections[di]
                if det.class_name != track.class_name:
                    continue
                score = iou(track.box, det.box)
                if score >= self.config.iou_match_threshold:
                    pairs.append((score, track_id, di))
        pairs.sort(key=lambda p: p[0], reverse=True)

        for _score, track_id, di in pairs:
            if track_id in matched_track_ids or di not in unmatched_detections:
                continue
            track = self.tracks[track_id]
            det = detections[di]
            moved = track.move_distance(det.box) >= self.config.move_threshold_px
            track.update_motion(
                det.box, now - track.last_seen,
                det.image_width or 1, det.image_height or 1,
            )
            track.box = det.box
            track.confidence = det.confidence
            track.last_seen = now
            track.misses = 0
            track.hits += 1
            matched_track_ids.add(track_id)
            unmatched_detections.remove(di)
            events.append(
                TrackEventRecord(
                    event=TrackEvent.MOVED if moved else TrackEvent.UPDATED,
                    track_id=track_id,
                    class_name=track.class_name,
                    box=track.box,
                    confidence=track.confidence,
                )
            )

        for di in unmatched_detections:
            det = detections[di]
            track_id = self._next_id(det.class_name)
            self.tracks[track_id] = Track(
                id=track_id,
                class_name=det.class_name,
                box=det.box,
                confidence=det.confidence,
                last_seen=now,
                first_seen=now,
            )
            # A track just created this frame was, by definition, observed
            # this frame — it must NOT be counted as a miss in the loop
            # below. (Bug found by test_brief_occlusion_does_not_immediately_
            # declare_lost: without this, a brand-new track got its first
            # "miss" charged against it in the very call that created it.)
            matched_track_ids.add(track_id)
            events.append(
                TrackEventRecord(
                    event=TrackEvent.ENTERED,
                    track_id=track_id,
                    class_name=det.class_name,
                    box=det.box,
                    confidence=det.confidence,
                )
            )

        to_delete = []
        for track_id, track in self.tracks.items():
            if track_id in matched_track_ids:
                continue
            track.misses += 1
            if track.misses >= self.config.max_misses_before_lost:
                events.append(
                    TrackEventRecord(
                        event=TrackEvent.LEFT,
                        track_id=track_id,
                        class_name=track.class_name,
                        box=track.box,
                        confidence=track.confidence,
                    )
                )
                to_delete.append(track_id)
            elif track.misses > 1:
                # First miss is tolerated silently (brief occlusion); from
                # the second miss on, tell callers this track is shaky.
                events.append(
                    TrackEventRecord(
                        event=TrackEvent.LOST,
                        track_id=track_id,
                        class_name=track.class_name,
                        box=track.box,
                        confidence=track.confidence,
                    )
                )

        for track_id in to_delete:
            del self.tracks[track_id]

        return events
