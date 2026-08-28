"""
Face detection and recognition of ENROLLED people only — Segment 8.

Uses OpenCV's bundled YuNet (detection, 232KB) and SFace (recognition, 37MB).
Both ship as ONNX and run on CPU in a few milliseconds, so no extra runtime
or heavyweight dependency is involved.

THE PRIVACY BOUNDARY, which is the entire point of this module:

    Detecting that a face exists is NOT the same as identifying whose it is.

Segment 7 already detects people without naming them, and that stays true for
everyone. This module can only ever produce a name for someone the user
DELIBERATELY enrolled — by choosing a photo and typing a name. There is no
path here that identifies a stranger, builds a face database from the camera
stream, or stores a face seen in passing. Faces observed during normal
operation are compared against enrolled templates and then discarded.

What gets stored is a 128-dimensional embedding, not a photograph.

MATCHING THRESHOLD: SFace's published same-identity threshold is 0.363 cosine.
That is used as the floor for a *possible* match, with a higher bar (0.45)
before the assistant is allowed to state the name without hedging — because
confidently telling a blind user "that's Rahul" when it is a stranger is a
far worse failure than saying "I think that might be Rahul".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("sfcc.perception.faces")

_WEIGHTS = Path(__file__).resolve().parents[3] / "models" / "weights"
DETECTOR_PATH = _WEIGHTS / "face_detection_yunet.onnx"
RECOGNIZER_PATH = _WEIGHTS / "face_recognition_sface.onnx"

# SFace's documented same-identity cosine threshold.
SAME_IDENTITY_THRESHOLD = 0.363
# Above this the agent may state the name plainly; between the two it must hedge.
CONFIDENT_IDENTITY_THRESHOLD = 0.45

DETECTION_SCORE_THRESHOLD = 0.85
NMS_THRESHOLD = 0.3
TOP_K = 20

# A face smaller than this is too low-resolution for a trustworthy embedding.
MIN_FACE_PIXELS = 40


@dataclass
class DetectedFace:
    """One face found in a frame. Carries no identity."""

    box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    raw: np.ndarray  # the YuNet row, needed by alignCrop

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def center_x(self) -> float:
        return (self.box[0] + self.box[2]) / 2


@dataclass
class FaceMatch:
    name: str
    similarity: float
    box: Tuple[float, float, float, float]

    @property
    def is_confident(self) -> bool:
        return self.similarity >= CONFIDENT_IDENTITY_THRESHOLD

    def payload(self) -> dict:
        return {
            "name": self.name,
            "similarity": round(self.similarity, 3),
            "is_confident": self.is_confident,
        }


class FaceEngine:
    """
    Detect faces and embed them. Degrades to unavailable (never raises) when
    the models are missing, so the rest of perception is unaffected.
    """

    def __init__(
        self,
        detector_path: Optional[Path] = None,
        recognizer_path: Optional[Path] = None,
    ) -> None:
        self.available = False
        self._detector = None
        self._recognizer = None
        detector_path = Path(detector_path or DETECTOR_PATH)
        recognizer_path = Path(recognizer_path or RECOGNIZER_PATH)

        if not detector_path.exists() or not recognizer_path.exists():
            logger.warning(
                "face models missing (%s / %s) — recognition disabled",
                detector_path.name, recognizer_path.name,
            )
            return
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(detector_path), "", (320, 320),
                DETECTION_SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K,
            )
            self._recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
            self.available = True
            logger.info("face engine ready (YuNet + SFace)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("face engine unavailable: %s", exc)
            self._detector = None
            self._recognizer = None

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        """Find faces. Returns an empty list rather than raising on failure."""
        if not self.available or frame is None or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        try:
            # YuNet requires the input size to be set before every detect call
            # whenever the frame size can change.
            self._detector.setInputSize((width, height))
            _retval, faces = self._detector.detect(frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning("face detection failed: %s", exc)
            return []
        if faces is None:
            return []

        results: List[DetectedFace] = []
        for row in faces:
            x, y, w, h = (float(v) for v in row[:4])
            if w < MIN_FACE_PIXELS or h < MIN_FACE_PIXELS:
                continue
            results.append(
                DetectedFace(box=(x, y, x + w, y + h), score=float(row[-1]), raw=row)
            )
        results.sort(key=lambda f: f.width, reverse=True)
        return results

    def embed(self, frame: np.ndarray, face: DetectedFace) -> Optional[np.ndarray]:
        """
        128-d L2-normalized embedding for one detected face.

        alignCrop applies the landmark-based alignment SFace was trained with;
        embedding an unaligned crop degrades accuracy badly.
        """
        if not self.available:
            return None
        try:
            aligned = self._recognizer.alignCrop(frame, face.raw)
            feature = self._recognizer.feature(aligned)
        except Exception as exc:  # noqa: BLE001
            logger.warning("face embedding failed: %s", exc)
            return None
        vector = np.asarray(feature, dtype=np.float32).flatten()
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else None

    def embed_largest(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Embed the most prominent face — the enrollment path, where the user
        supplies a photo of one specific person.
        """
        faces = self.detect(frame)
        if not faces:
            return None
        return self.embed(frame, faces[0])

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity. Both vectors are already unit length."""
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        return float(np.dot(a, b))

    def match(
        self,
        embedding: np.ndarray,
        enrolled: Sequence[Tuple[str, np.ndarray]],
        box: Tuple[float, float, float, float],
    ) -> Optional[FaceMatch]:
        """
        Best enrolled identity for one face, or None if nothing clears the
        same-identity threshold. Returning None is the common, correct answer:
        most faces the camera sees belong to nobody the user enrolled.
        """
        best_name, best_score = None, 0.0
        for name, candidate in enrolled:
            score = self.similarity(embedding, candidate)
            if score > best_score:
                best_name, best_score = name, score
        if best_name is None or best_score < SAME_IDENTITY_THRESHOLD:
            return None
        return FaceMatch(name=best_name, similarity=best_score, box=box)
