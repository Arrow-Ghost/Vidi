"""
World model — Segment 4.

The continuously-updated store of "what's currently around the user,"
built from ObjectTracker events plus per-track DirectionEstimator /
SmoothedDistanceEstimator state. The agent (Segment 12+) queries this, not
the detector or tracker directly — this is the seam that lets perception
internals change without the agent noticing.

summarize() is the "don't narrate everything" requirement made concrete:
it prioritizes requested object > people > obstacles > everything else,
and caps how many entities it ever returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from app.perception.depth import DepthMap, FusedDistanceEstimator
from app.perception.direction import DirectionEstimate, DirectionEstimator
from app.perception.distance import DistanceEstimate
from app.perception.object_detector import Detection
from app.perception.tracker import ObjectTracker, TrackEvent, TrackEventRecord


@dataclass
class Entity:
    id: str
    type: str
    direction: DirectionEstimate
    distance: DistanceEstimate
    confidence: float  # detector confidence — distinct from direction/distance confidence
    is_person: bool = False


# Things a user could physically bump into — ranked above generic "other"
# objects but below people and whatever the user explicitly asked about.
_OBSTACLE_CLASSES = {"chair", "dining table", "suitcase"}

_PRIORITY_REQUESTED = 0
_PRIORITY_PERSON = 1
_PRIORITY_OBSTACLE = 2
_PRIORITY_OTHER = 3


class WorldModel:
    def __init__(self) -> None:
        self.entities: Dict[str, Entity] = {}
        self._direction_estimators: Dict[str, DirectionEstimator] = {}
        self._distance_estimators: Dict[str, FusedDistanceEstimator] = {}

    def apply_frame(
        self,
        tracker: ObjectTracker,
        detections: List[Detection],
        image_width: int,
        image_height: int,
        depth_map: Optional[DepthMap] = None,
    ) -> List[TrackEventRecord]:
        """
        Runs one frame's detections through the tracker and refreshes
        entity state for every track still alive afterward. Returns the
        raw tracker events too — the attention manager (Segment 27) needs
        those to decide what's actually worth mentioning out loud.
        """
        events = tracker.update(detections)

        for event in events:
            if event.event == TrackEvent.LEFT:
                self.entities.pop(event.track_id, None)
                self._direction_estimators.pop(event.track_id, None)
                self._distance_estimators.pop(event.track_id, None)
                continue

            direction_estimator = self._direction_estimators.setdefault(
                event.track_id, DirectionEstimator()
            )
            distance_estimator = self._distance_estimators.setdefault(
                event.track_id, FusedDistanceEstimator()
            )

            direction = direction_estimator.update(event.box.center_x, image_width)
            # Passing the whole box (not just its height) lets the estimator
            # sample the depth map at this object's actual location.
            distance = distance_estimator.update(
                event.class_name, event.box, image_width, image_height, depth_map
            )

            self.entities[event.track_id] = Entity(
                id=event.track_id,
                type=event.class_name,
                direction=direction,
                distance=distance,
                confidence=event.confidence,
                is_person=event.class_name == "person",
            )

        return events

    def summarize(self, requested_class: Optional[str] = None, max_items: int = 4) -> List[Entity]:
        """Prioritized, capped subset of entities — never dumps everything tracked."""

        def priority(entity: Entity) -> int:
            if requested_class and entity.type == requested_class:
                return _PRIORITY_REQUESTED
            if entity.is_person:
                return _PRIORITY_PERSON
            if entity.type in _OBSTACLE_CLASSES:
                return _PRIORITY_OBSTACLE
            return _PRIORITY_OTHER

        ranked = sorted(self.entities.values(), key=lambda e: (priority(e), -e.confidence))
        return ranked[:max_items]

    def find(self, class_name: str) -> List[Entity]:
        return [e for e in self.entities.values() if e.type == class_name]
