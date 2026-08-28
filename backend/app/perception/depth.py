"""
Monocular depth — Segment 6, real model.

Runs Depth Anything V2 Small (ViT-S) on the Intel iGPU. Measured on this
machine (see docs/PERFORMANCE.md):

    224x224   47.3 ms   21.2 FPS   <-- default
    308x308  107.0 ms    9.3 FPS
    392x392  204.7 ms    4.9 FPS

WHAT THIS MODEL ACTUALLY GIVES YOU — and what it does not.

Depth Anything outputs *relative inverse depth*: larger values are nearer,
but the units are arbitrary and the scale drifts between frames. It is NOT a
range finder. Converting its raw output into "3.2 meters" would be inventing
precision that does not exist, which is the single most dangerous thing this
system could do.

So depth is used for the two things it is genuinely reliable at:

  1. ORDERING — "the chair is nearer than the table" is trustworthy.
  2. CROSS-CHECKING the bbox-height size prior.

The size prior (an object's apparent height vs. how big that class usually
is) carries weak absolute information; depth carries strong relative
information. Fusing them:

  * both agree            -> report that range with RAISED confidence
  * adjacent disagreement -> report the depth-derived range, lower confidence
  * gross disagreement    -> report UNKNOWN

That last branch matters most. When the two independent estimates contradict
each other, the honest answer is "I can tell you where it is but not how far",
and that is exactly what the agent then says.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.perception.distance import (
    DistanceEstimate,
    DistanceRange,
    _RANGE_LABELS,
    DistanceEstimator,
)

logger = logging.getLogger("vidi.perception.depth")

DEFAULT_DEPTH_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "weights" / "depth_anything_v2_small.onnx"
)

# Must be a multiple of 14 — the ViT patch size. 224 = 14*16.
DEPTH_INPUT_SIZE = 224

# Depth Anything was trained with ImageNet normalization.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Percentile-rank thresholds (1.0 = nearest thing in frame) mapping relative
# depth onto the qualitative ranges. Deliberately coarse.
_DEPTH_RANGE_THRESHOLDS: List[Tuple[float, DistanceRange]] = [
    (0.86, DistanceRange.VERY_CLOSE),
    (0.62, DistanceRange.CLOSE),
    (0.33, DistanceRange.NEARBY),
    (0.00, DistanceRange.FAR),
]

_RANGE_ORDER = [
    DistanceRange.VERY_CLOSE,
    DistanceRange.CLOSE,
    DistanceRange.NEARBY,
    DistanceRange.FAR,
]

# Confidences for each fusion outcome. Never 1.0 — this is a monocular
# estimate and must not present itself as measurement.
CONF_AGREE = 0.78
CONF_ADJACENT = 0.55
CONF_CONFLICT = 0.15  # below the assert floor -> reported as UNKNOWN


@dataclass
class DepthMap:
    """One frame's relative inverse-depth map, plus when it was produced."""

    values: np.ndarray  # (H, W) float32; larger = nearer
    created_at: float
    inference_ms: float

    def age_s(self, now: Optional[float] = None) -> float:
        return (now if now is not None else time.time()) - self.created_at


class DepthEstimator:
    """
    Wraps the depth model. Returns None (never raises, never fabricates) if
    the model is unavailable, so the system degrades to the size-prior
    heuristic rather than losing distance entirely.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        input_size: int = DEPTH_INPUT_SIZE,
        device: str = "GPU",
    ) -> None:
        self.input_size = input_size
        self.available = False
        self.last_inference_ms: Optional[float] = None
        self._compiled = None
        path = Path(model_path) if model_path else DEFAULT_DEPTH_PATH

        if not path.exists():
            logger.warning("depth model missing at %s — depth disabled", path)
            return
        try:
            import openvino as ov

            core = ov.Core()
            target = device if device in core.available_devices else "CPU"
            model = core.read_model(str(path))
            # Static shape: the graph is dynamic, and recompiling per frame
            # size would cost far more than the inference itself.
            model.reshape({model.input(0): [1, 3, input_size, input_size]})
            self._compiled = core.compile_model(model, target, {"PERFORMANCE_HINT": "LATENCY"})
            self.device = target
            self.available = True
            logger.info("depth model ready on %s (%dx%d)", target, input_size, input_size)
        except Exception as exc:  # noqa: BLE001
            logger.warning("depth model unavailable: %s", exc)
            self._compiled = None

    def infer(self, frame: np.ndarray) -> Optional[DepthMap]:
        if not self.available or self._compiled is None:
            return None
        try:
            started = time.perf_counter()
            resized = cv2.resize(
                frame, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch = np.transpose(rgb, (2, 0, 1))
            batch = (batch - _IMAGENET_MEAN) / _IMAGENET_STD
            batch = np.ascontiguousarray(batch[np.newaxis, ...])

            result = self._compiled(batch)
            depth = np.asarray(result[self._compiled.output(0)]).squeeze()
            elapsed = (time.perf_counter() - started) * 1000
            self.last_inference_ms = elapsed
            return DepthMap(values=depth.astype(np.float32),
                            created_at=time.time(), inference_ms=elapsed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("depth inference failed: %s", exc)
            return None


def sample_box_depth(
    depth_map: DepthMap, box, image_width: int, image_height: int
) -> Optional[float]:
    """
    Robust relative-depth reading for one bounding box.

    Samples only the central half of the box: box edges routinely include
    background pixels, and for a thin object like a chair leg the background
    would dominate a naive mean. Median, not mean, for the same reason.
    """
    depth = depth_map.values
    dh, dw = depth.shape[:2]
    if image_width <= 0 or image_height <= 0:
        return None

    # Box is in original-frame pixels; the depth map is its own resolution.
    cx = box.center_x / image_width
    cy = box.center_y / image_height
    half_w = (box.width / image_width) * 0.25
    half_h = (box.height / image_height) * 0.25

    x1 = int(np.clip((cx - half_w) * dw, 0, dw - 1))
    x2 = int(np.clip((cx + half_w) * dw, 0, dw - 1))
    y1 = int(np.clip((cy - half_h) * dh, 0, dh - 1))
    y2 = int(np.clip((cy + half_h) * dh, 0, dh - 1))
    if x2 <= x1:
        x2 = min(x1 + 1, dw - 1)
    if y2 <= y1:
        y2 = min(y1 + 1, dh - 1)

    patch = depth[y1 : y2 + 1, x1 : x2 + 1]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def depth_percentile(depth_map: DepthMap, value: float) -> float:
    """
    Where this reading sits within the frame's own depth distribution
    (0 = farthest thing visible, 1 = nearest). Self-normalizing, which is
    what makes it robust to the model's arbitrary and drifting scale.
    """
    flat = depth_map.values.reshape(-1)
    if flat.size == 0:
        return 0.5
    # Subsample: a 224x224 map is 50k values and exact ranking is unnecessary.
    sample = flat[:: max(1, flat.size // 4096)]
    return float(np.mean(sample <= value))


def _range_from_percentile(percentile: float) -> DistanceRange:
    for threshold, range_ in _DEPTH_RANGE_THRESHOLDS:
        if percentile >= threshold:
            return range_
    return DistanceRange.FAR


def fuse(
    depth_range: DistanceRange, prior: DistanceEstimate, method_suffix: str = ""
) -> DistanceEstimate:
    """
    Combine the depth-derived range with the size-prior range.

    Conflict produces UNKNOWN on purpose: two independent estimates that
    disagree badly are evidence that neither should be spoken aloud.
    """
    if prior.range == DistanceRange.UNKNOWN:
        # No usable prior — trust depth alone, but only at medium confidence.
        return DistanceEstimate(
            range=depth_range,
            confidence=CONF_ADJACENT,
            method="depth_anything_v2" + method_suffix,
            label=_RANGE_LABELS[depth_range],
        )

    depth_index = _RANGE_ORDER.index(depth_range)
    prior_index = _RANGE_ORDER.index(prior.range)
    gap = abs(depth_index - prior_index)

    if gap == 0:
        return DistanceEstimate(
            range=depth_range, confidence=CONF_AGREE,
            method="depth+size_prior_agree" + method_suffix,
            label=_RANGE_LABELS[depth_range],
        )
    if gap == 1:
        return DistanceEstimate(
            range=depth_range, confidence=CONF_ADJACENT,
            method="depth+size_prior_adjacent" + method_suffix,
            label=_RANGE_LABELS[depth_range],
        )
    return DistanceEstimate(
        range=DistanceRange.UNKNOWN, confidence=CONF_CONFLICT,
        method="depth+size_prior_conflict" + method_suffix,
        label=_RANGE_LABELS[DistanceRange.UNKNOWN],
    )


class FusedDistanceEstimator:
    """
    Drop-in replacement for SmoothedDistanceEstimator that additionally uses a
    real depth map when one is available. Keeps the same per-track lifecycle.
    """

    def __init__(self, history_size: int = 5) -> None:
        self._prior = DistanceEstimator()
        self._history: List[DistanceRange] = []
        self._history_size = history_size

    def update(
        self,
        class_name: str,
        box,
        image_width: int,
        image_height: int,
        depth_map: Optional[DepthMap] = None,
        max_depth_age_s: float = 1.5,
    ) -> DistanceEstimate:
        prior = self._prior.estimate(class_name, box.height, image_height)

        estimate = prior
        if depth_map is not None and depth_map.age_s() <= max_depth_age_s:
            reading = sample_box_depth(depth_map, box, image_width, image_height)
            if reading is not None:
                percentile = depth_percentile(depth_map, reading)
                estimate = fuse(_range_from_percentile(percentile), prior)

        # Temporal smoothing over the fused result: one noisy frame must not
        # redefine the answer the user hears.
        self._history.append(estimate.range)
        del self._history[: -self._history_size]
        smoothed = max(set(self._history), key=self._history.count)

        if smoothed != estimate.range:
            # Majority history disagrees with this frame — report the stable
            # value but do not inherit this frame's confidence.
            return DistanceEstimate(
                range=smoothed,
                confidence=min(estimate.confidence, CONF_ADJACENT),
                method=estimate.method + "+smoothed",
                label=_RANGE_LABELS[smoothed],
            )
        return DistanceEstimate(
            range=smoothed, confidence=estimate.confidence,
            method=estimate.method + "+smoothed", label=_RANGE_LABELS[smoothed],
        )

    def reset(self) -> None:
        self._history.clear()
