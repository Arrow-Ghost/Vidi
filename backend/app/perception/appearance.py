"""
Appearance embeddings for enrolled belongings — Segment 8 (objects half).

YOLO can tell you there is *a* backpack in view. It cannot tell you whether
it is *your* backpack. This module closes that gap: the user enrolls a photo
of their own bag, and detected backpacks are then compared against it.

HOW IT WORKS
MobileNetV2 (13MB ONNX) runs on the same OpenVINO iGPU path as the detector.
The penultimate pooled layer gives a 1280-d appearance descriptor that
captures colour, texture and shape. Cosine similarity against the enrolled
descriptor gives the match.

WHAT THIS CAN AND CANNOT DO — stated plainly because the honest limits matter:

  Works well:  telling an enrolled item from a clearly different object, and
               re-recognizing the same item under changed lighting, blur,
               scale or JPEG compression (measured floor 0.857).
  Works badly: two items of the SAME class that merely differ in colour. Deep
               ImageNet features deliberately suppress colour, and the
               measured separation margin is only ~0.11 even between wholly
               different objects. Within-class discrimination is the weak
               case and the system must hedge there, never assert.

So the class match is a HARD gate (a chair is never your backpack), and the
similarity score is reported with an explicit confidence band. Below the
confident threshold the agent is required to hedge — "that might be yours" —
rather than assert. Getting this wrong means walking a blind user over to a
stranger's bag.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("vidi.perception.appearance")

MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "weights" / "mobilenetv2_embed.onnx"
)
INPUT_SIZE = 224

# ImageNet normalization — MobileNetV2's training preprocessing.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Cosine thresholds, CALIBRATED FROM MEASUREMENT rather than picked by feel.
#
# Measured over real crops (bus, two people, a player, grass) each compared
# against brightness / blur / scale / JPEG-quality variants of itself:
#
#     same object, worst case      0.857
#     different objects, best case 0.747
#
# So the usable margin is only ~0.11, and an earlier guess of 0.62 would have
# reported two completely different objects as a match. POSSIBLE sits above
# the different-object ceiling; CONFIDENT sits just under the same-object
# floor. Anything between them is reported as "possible" and MUST be hedged.
#
# A colour histogram was tried as a fusion term and measurably made this
# WORSE at every weight (margin 0.110 -> 0.086 -> 0.059 -> 0.032 as colour
# weight rose), so it is deliberately not used.
POSSIBLE_MATCH_THRESHOLD = 0.80
CONFIDENT_MATCH_THRESHOLD = 0.86

# Crops smaller than this carry too little detail to compare meaningfully.
MIN_CROP_PIXELS = 32


def _find_feature_tensor(path: Path) -> Optional[str]:
    """
    Name of the tensor feeding the final classifier.

    We want the 1280-d pooled FEATURE vector, not the 1000-d class logits.
    Logits are a poor appearance descriptor: two different objects that both
    look vaguely like several ImageNet classes produce highly correlated
    logits, and measured on this machine that put a bus and a person at 0.708
    cosine — indistinguishable at any usable threshold. The pooled features
    separate them properly.

    Found by walking the graph rather than hardcoding a node name, so this
    keeps working if the model file is swapped for another classifier.
    """
    try:
        import onnx

        graph = onnx.load(str(path)).graph
        for node in reversed(graph.node):
            if node.op_type in ("Gemm", "MatMul") and node.input:
                return node.input[0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not locate feature tensor: %s", exc)
    return None


class AppearanceEmbedder:
    def __init__(self, model_path: Optional[Path] = None, device: str = "GPU") -> None:
        self.available = False
        self._compiled = None
        self._output = None
        self.using_features = False
        path = Path(model_path or MODEL_PATH)
        if not path.exists():
            logger.warning("appearance model missing at %s — belongings matching disabled", path)
            return
        try:
            import openvino as ov

            core = ov.Core()
            target = device if device in core.available_devices else "CPU"
            model = core.read_model(str(path))
            model.reshape({model.input(0): [1, 3, INPUT_SIZE, INPUT_SIZE]})

            feature_tensor = _find_feature_tensor(path)
            if feature_tensor:
                try:
                    model.add_outputs([feature_tensor])
                    self.using_features = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "could not expose feature tensor %s (%s); falling back to logits",
                        feature_tensor, exc,
                    )

            self._compiled = core.compile_model(model, target, {"PERFORMANCE_HINT": "LATENCY"})
            # The added feature tensor becomes the LAST output; index 0 stays
            # the original classifier head.
            self._output = (
                self._compiled.output(len(self._compiled.outputs) - 1)
                if self.using_features else self._compiled.output(0)
            )
            self.device = target
            self.available = True
            logger.info(
                "appearance embedder ready on %s (features=%s)", target, self.using_features
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("appearance embedder unavailable: %s", exc)
            self._compiled = None

    def embed(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """L2-normalized appearance descriptor for one image crop."""
        if not self.available or crop is None or crop.size == 0:
            return None
        height, width = crop.shape[:2]
        if height < MIN_CROP_PIXELS or width < MIN_CROP_PIXELS:
            return None
        try:
            resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch = np.transpose(rgb, (2, 0, 1))
            batch = (batch - _MEAN) / _STD
            batch = np.ascontiguousarray(batch[np.newaxis, ...])
            raw = np.asarray(self._compiled(batch)[self._output]).flatten()
        except Exception as exc:  # noqa: BLE001
            logger.warning("appearance embedding failed: %s", exc)
            return None
        norm = np.linalg.norm(raw)
        return (raw / norm).astype(np.float32) if norm > 0 else None

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        return float(np.dot(a, b))

    def best_match(
        self, embedding: np.ndarray, enrolled: Sequence[Tuple[str, np.ndarray]]
    ) -> Tuple[Optional[str], float]:
        """Highest-scoring enrolled belonging, with its raw similarity."""
        best_name, best_score = None, 0.0
        for name, candidate in enrolled:
            score = self.similarity(embedding, candidate)
            if score > best_score:
                best_name, best_score = name, score
        return best_name, best_score


def crop_from_box(frame: np.ndarray, box, padding: float = 0.05) -> Optional[np.ndarray]:
    """
    Crop a detection out of the frame, with a little padding for context.
    Returns None when the box is degenerate rather than an empty array.
    """
    if frame is None or frame.size == 0:
        return None
    height, width = frame.shape[:2]
    pad_x = box.width * padding
    pad_y = box.height * padding
    x1 = int(max(0, box.x1 - pad_x))
    y1 = int(max(0, box.y1 - pad_y))
    x2 = int(min(width, box.x2 + pad_x))
    y2 = int(min(height, box.y2 + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def describe_match(score: float) -> str:
    """Confidence band, used to force the agent to hedge appropriately."""
    if score >= CONFIDENT_MATCH_THRESHOLD:
        return "confident"
    if score >= POSSIBLE_MATCH_THRESHOLD:
        return "possible"
    return "no_match"
