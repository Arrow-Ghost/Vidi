"""
Correctness gate for the hand-rolled ONNX detection path.

The system runs YOLO through raw ONNX (OpenVINO/ONNXRuntime) rather than the
ultralytics wrapper, because that is 400x faster on this hardware. The risk
of doing that is silently wrong letterboxing or NMS: boxes that are offset,
mis-scaled, or assigned the wrong class. For a blind user that is not a
cosmetic bug — it means being told an obstacle is on the wrong side.

So these tests assert PARITY against ultralytics' own output on a real
photograph, not merely that our pipeline is self-consistent.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.perception.postprocess import letterbox, postprocess, preprocess
from app.perception.runtime import select_backend

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_ONNX = ROOT / "models" / "weights" / "yolov8n.onnx"
WEIGHTS_PT = ROOT / "models" / "weights" / "yolov8n.pt"
BUS = Path(__file__).parent / "fixtures" / "bus.jpg"

IMGSZ = 480


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def test_letterbox_preserves_aspect_ratio_and_centers():
    img = np.zeros((1080, 810, 3), dtype=np.uint8)
    padded, info = letterbox(img, IMGSZ)
    assert padded.shape == (IMGSZ, IMGSZ, 3)
    # 1080 is the long side, so scale is bound by height.
    assert info.scale == pytest.approx(IMGSZ / 1080)
    # Padding goes on the short side (width) only.
    assert info.pad_y == pytest.approx(0.0)
    assert info.pad_x > 0


def test_preprocess_produces_normalized_nchw_batch():
    img = cv2.imread(str(BUS))
    batch, _ = preprocess(img, IMGSZ)
    assert batch.shape == (1, 3, IMGSZ, IMGSZ)
    assert batch.dtype == np.float32
    assert 0.0 <= batch.min() and batch.max() <= 1.0


@pytest.mark.skipif(not WEIGHTS_ONNX.exists(), reason="ONNX weights not exported")
def test_onnx_pipeline_matches_ultralytics_reference():
    """
    THE important test: our fast path must find the same objects, in the
    same places, as the reference implementation.
    """
    ultralytics = pytest.importorskip("ultralytics")
    img = cv2.imread(str(BUS))
    assert img is not None

    # --- our fast path ---
    backend = select_backend(WEIGHTS_ONNX)
    batch, info = preprocess(img, IMGSZ)
    ours = postprocess(backend.infer(batch), info, img.shape[:2], conf_threshold=0.45)

    # --- reference path ---
    reference_model = ultralytics.YOLO(str(WEIGHTS_PT))
    ref_result = reference_model.predict(img, imgsz=IMGSZ, conf=0.45, verbose=False)[0]
    reference = [
        (ref_result.names[int(b.cls[0])], [float(v) for v in b.xyxy[0]])
        for b in ref_result.boxes
    ]

    assert len(ours) == len(reference), (
        f"detection count differs: ours={[d.class_name for d in ours]} "
        f"reference={[r[0] for r in reference]}"
    )

    # Every reference detection must have a matching detection from us:
    # same class, and a box overlapping >0.9 IoU.
    unmatched = []
    for ref_name, ref_box in reference:
        if not any(
            d.class_name == ref_name and _iou(d.xyxy, ref_box) > 0.9 for d in ours
        ):
            unmatched.append((ref_name, ref_box))
    assert not unmatched, f"reference detections we failed to reproduce: {unmatched}"


@pytest.mark.skipif(not WEIGHTS_ONNX.exists(), reason="ONNX weights not exported")
def test_empty_scene_returns_no_detections_rather_than_guessing():
    """A blank frame must produce zero detections — never a low-confidence guess."""
    backend = select_backend(WEIGHTS_ONNX)
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    batch, info = preprocess(blank, IMGSZ)
    assert postprocess(backend.infer(batch), info, blank.shape[:2], conf_threshold=0.45) == []
