"""
Recognition of enrolled people and belongings — Segment 8.

Combines the face engine and the appearance embedder against what the user
has actually enrolled, and answers one question per detection: is this
someone/something the user told me about?

TWO HARD RULES, both enforced here rather than left to the prompt:

1. NOBODY is identified unless the user enrolled them. A face that matches
   nothing enrolled is reported as "a person", exactly as before — never
   described, never characterized.

2. A belonging must match on CLASS as well as appearance. A chair that happens
   to share a colour histogram with a backpack can never be reported as the
   user's backpack, because the detector class is a hard gate.

Every result carries a confidence band ("confident" / "possible") so the agent
is forced to hedge when the evidence is weak. Walking a blind user across a
room to a stranger's bag is the failure this is built to avoid.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.perception.appearance import (
    AppearanceEmbedder,
    CONFIDENT_MATCH_THRESHOLD,
    POSSIBLE_MATCH_THRESHOLD,
    crop_from_box,
    describe_match,
)
from app.perception.direction import DirectionBucket, _raw_bucket
from app.perception.faces import FaceEngine, FaceMatch

logger = logging.getLogger("sfcc.perception.recognition")

# Enrolled embeddings are re-read from the database at most this often, so an
# enrollment made in the UI takes effect within a couple of seconds without
# hitting SQLite on every frame.
ENROLLMENT_CACHE_TTL_S = 3.0


@dataclass
class PersonIdentification:
    name: str
    similarity: float
    is_confident: bool
    direction: DirectionBucket

    def payload(self) -> Dict:
        return {
            "name": self.name,
            "similarity": round(self.similarity, 3),
            "is_confident": self.is_confident,
            "direction": self.direction.value,
        }


@dataclass
class BelongingIdentification:
    name: str
    object_class: str
    similarity: float
    band: str                 # confident | possible
    direction: DirectionBucket
    entity_id: Optional[str] = None

    def payload(self) -> Dict:
        return {
            "name": self.name,
            "object_class": self.object_class,
            "similarity": round(self.similarity, 3),
            "match_band": self.band,
            "is_confident": self.band == "confident",
            "direction": self.direction.value,
            "entity_id": self.entity_id,
        }


class RecognitionEngine:
    def __init__(
        self,
        store=None,
        faces: Optional[FaceEngine] = None,
        appearance: Optional[AppearanceEmbedder] = None,
    ) -> None:
        self.store = store
        self.faces = faces if faces is not None else FaceEngine()
        self.appearance = appearance if appearance is not None else AppearanceEmbedder()
        self._cache: Dict[str, List[Tuple[str, np.ndarray, Optional[str]]]] = {}
        self._cached_at = 0.0

    @property
    def faces_available(self) -> bool:
        return self.faces.available

    @property
    def belongings_available(self) -> bool:
        return self.appearance.available

    # ---------------- enrolled data ----------------

    def _enrolled(self, kind: str) -> List[Tuple[str, np.ndarray, Optional[str]]]:
        now = time.time()
        if now - self._cached_at > ENROLLMENT_CACHE_TTL_S:
            self._cache = {"person": [], "belonging": []}
            if self.store is not None:
                for record in self.store.list_enrollments():
                    self._cache.setdefault(record.kind, []).append(
                        (record.name, record.vector(), record.object_class)
                    )
            self._cached_at = now
        return self._cache.get(kind, [])

    def invalidate_cache(self) -> None:
        """Force a reload — called right after an enrollment changes."""
        self._cached_at = 0.0

    def enrolled_names(self, kind: str) -> List[str]:
        return [name for name, _vector, _cls in self._enrolled(kind)]

    # ---------------- enrollment ----------------

    def embed_face_from_photo(self, image: np.ndarray) -> Optional[np.ndarray]:
        return self.faces.embed_largest(image)

    def embed_object_from_photo(self, image: np.ndarray) -> Optional[np.ndarray]:
        return self.appearance.embed(image)

    # ---------------- recognition ----------------

    def identify_people(self, frame: np.ndarray) -> Tuple[List[PersonIdentification], int]:
        """
        Returns (identified people, total faces seen).

        The second number matters: three faces with one match means two people
        the user has NOT enrolled, and the agent must not describe them.
        """
        if not self.faces_available or frame is None:
            return [], 0
        enrolled = [(name, vector) for name, vector, _cls in self._enrolled("person")]
        detected = self.faces.detect(frame)
        if not detected:
            return [], 0
        if not enrolled:
            return [], len(detected)

        width = frame.shape[1]
        identified: List[PersonIdentification] = []
        for face in detected:
            embedding = self.faces.embed(frame, face)
            if embedding is None:
                continue
            match: Optional[FaceMatch] = self.faces.match(embedding, enrolled, face.box)
            if match is None:
                continue
            identified.append(
                PersonIdentification(
                    name=match.name,
                    similarity=match.similarity,
                    is_confident=match.is_confident,
                    direction=_raw_bucket(face.center_x / width) if width else DirectionBucket.CENTER,
                )
            )
        identified.sort(key=lambda p: p.similarity, reverse=True)
        return identified, len(detected)

    def identify_belongings(
        self, frame: np.ndarray, detections: Sequence, entity_ids: Optional[Dict[int, str]] = None
    ) -> List[BelongingIdentification]:
        """
        Match detected objects against enrolled belongings.

        `detections` are YOLO detections; the class name is used as a hard
        gate before any appearance comparison happens.
        """
        if not self.belongings_available or frame is None or not detections:
            return []
        enrolled = self._enrolled("belonging")
        if not enrolled:
            return []

        width = frame.shape[1]
        results: List[BelongingIdentification] = []
        for index, detection in enumerate(detections):
            # HARD GATE: only compare against belongings enrolled as this class.
            candidates = [
                (name, vector) for name, vector, cls in enrolled
                if cls is None or cls == detection.class_name
            ]
            if not candidates:
                continue
            crop = crop_from_box(frame, detection.box)
            if crop is None:
                continue
            embedding = self.appearance.embed(crop)
            if embedding is None:
                continue
            name, score = self.appearance.best_match(embedding, candidates)
            if name is None or score < POSSIBLE_MATCH_THRESHOLD:
                continue
            results.append(
                BelongingIdentification(
                    name=name,
                    object_class=detection.class_name,
                    similarity=score,
                    band=describe_match(score),
                    direction=(
                        _raw_bucket(detection.box.center_x / width)
                        if width else DirectionBucket.CENTER
                    ),
                    entity_id=(entity_ids or {}).get(index),
                )
            )
        results.sort(key=lambda b: b.similarity, reverse=True)
        return results
