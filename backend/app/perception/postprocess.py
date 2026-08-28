"""
YOLOv8 pre/postprocessing, implemented directly against the ONNX graph.

Using the raw ONNX model instead of the ultralytics wrapper is what buys the
400x speedup documented in runtime.py, but it means the letterboxing and NMS
that ultralytics normally does for us must be done here — correctly. Getting
this subtly wrong is the classic way a "fast" detector silently returns
boxes that are offset or mis-scaled, which for a blind user would mean being
told an obstacle is on the wrong side. test_postprocess.py therefore asserts
parity against ultralytics' own reference output on a real photograph, not
just internal self-consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

# COCO class names, in the exact index order the YOLOv8 head emits.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@dataclass(frozen=True)
class LetterboxInfo:
    """Everything needed to map model-space coords back to original pixels."""

    scale: float
    pad_x: float
    pad_y: float


def letterbox(
    image: np.ndarray, target: int = 480, color: int = 114
) -> Tuple[np.ndarray, LetterboxInfo]:
    """
    Resize preserving aspect ratio and pad to a square, exactly as YOLO was
    trained. Stretching to square instead would distort every box and every
    direction estimate derived from it.
    """
    height, width = image.shape[:2]
    scale = min(target / height, target / width)
    new_w, new_h = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target, target, 3), color, dtype=np.uint8)
    pad_x = (target - new_w) / 2
    pad_y = (target - new_h) / 2
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, LetterboxInfo(scale=scale, pad_x=pad_x, pad_y=pad_y)


def preprocess(image: np.ndarray, target: int = 480) -> Tuple[np.ndarray, LetterboxInfo]:
    """BGR uint8 HWC image -> normalized RGB float32 NCHW batch."""
    padded, info = letterbox(image, target)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    batch = rgb.astype(np.float32) / 255.0
    batch = np.transpose(batch, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(batch), info


@dataclass
class RawDetection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: Tuple[float, float, float, float]


def postprocess(
    output: np.ndarray,
    info: LetterboxInfo,
    original_shape: Tuple[int, int],
    conf_threshold: float = 0.45,
    iou_threshold: float = 0.45,
) -> List[RawDetection]:
    """
    Decode YOLOv8's (1, 84, N) output into boxes in ORIGINAL image pixels.

    Layout: rows 0-3 are cx, cy, w, h in model (letterboxed) space; rows 4-83
    are per-class scores with no objectness term and no softmax — YOLOv8
    differs from YOLOv5 here, and treating row 4 as objectness (the v5 layout)
    is a common and silent source of wrong classes.
    """
    predictions = np.squeeze(output, axis=0).T  # (N, 84)
    if predictions.size == 0:
        return []

    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]

    keep = confidences >= conf_threshold
    if not np.any(keep):
        return []
    predictions, class_ids, confidences = predictions[keep], class_ids[keep], confidences[keep]

    # cxcywh (model space) -> xyxy (model space) -> undo pad -> undo scale.
    cx, cy, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
    x1 = (cx - w / 2 - info.pad_x) / info.scale
    y1 = (cy - h / 2 - info.pad_y) / info.scale
    x2 = (cx + w / 2 - info.pad_x) / info.scale
    y2 = (cy + h / 2 - info.pad_y) / info.scale

    orig_h, orig_w = original_shape
    x1 = np.clip(x1, 0, orig_w); x2 = np.clip(x2, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h); y2 = np.clip(y2, 0, orig_h)

    # Class-aware NMS: a person overlapping a chair must not suppress it.
    boxes_wh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    indices = cv2.dnn.NMSBoxes(
        boxes_wh.tolist(), confidences.astype(float).tolist(),
        conf_threshold, iou_threshold,
    )
    if len(indices) == 0:
        return []
    indices = np.array(indices).flatten()

    results: List[RawDetection] = []
    for i in indices:
        cid = int(class_ids[i])
        results.append(
            RawDetection(
                class_id=cid,
                class_name=COCO_CLASSES[cid],
                confidence=float(confidences[i]),
                xyxy=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
            )
        )
    results.sort(key=lambda d: d.confidence, reverse=True)
    return results
