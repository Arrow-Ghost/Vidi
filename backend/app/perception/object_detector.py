"""
ObjectDetector — Segment 3.

Runs YOLOv8n over the fastest inference backend available on this machine
(see runtime.py for the measured numbers that drove that choice: OpenVINO on
the Intel iGPU at ~9ms/frame, versus ~3600ms through torch). The ultralytics
wrapper is used only for one-time ONNX export, never on the hot path.

Every detection carries a real confidence score; nothing here ever asserts an
object exists without one, and nothing below the configured threshold is
returned.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from app.perception.postprocess import postprocess, preprocess
from app.perception.runtime import InferenceBackend, select_backend

logger = logging.getLogger("vidi.perception.detector")

# Everyday objects relevant to an assistive agent (project brief's minimum
# list), mapped onto actual COCO class names where they differ.
# "door", "stairs", "keys", "wallet", "plate" are NOT COCO classes and have
# no pretrained detector head here — see ARCHITECTURE.md's Segment 3
# limitations for how each is handled instead of silently pretending they're
# detected. They are routed to the open-vocabulary visual grounding tool.
RELEVANT_CLASSES = {
    "person", "chair", "dining table", "backpack", "handbag", "suitcase",
    "bottle", "cell phone", "laptop", "book", "cup", "bowl", "banana",
    "apple", "sandwich", "car", "bus", "truck", "bicycle", "motorcycle",
    "couch", "bed", "tv", "keyboard", "mouse", "remote", "potted plant",
    "traffic light", "stop sign", "bench", "umbrella", "clock", "sink",
    "refrigerator", "microwave", "oven", "toilet",
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.45
# Matches the exported ONNX graph's fixed input size. Changing this requires
# re-exporting the model, so it is not a free runtime knob.
DEFAULT_IMAGE_SIZE = 480

_WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "models" / "weights"
DEFAULT_ONNX_PATH = _WEIGHTS_DIR / "yolov8n.onnx"
DEFAULT_PT_PATH = _WEIGHTS_DIR / "yolov8n.pt"


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Detection:
    class_name: str
    confidence: float
    box: BoundingBox
    timestamp: float
    image_width: int
    image_height: int


def ensure_onnx_export(
    pt_path: Path = DEFAULT_PT_PATH,
    onnx_path: Path = DEFAULT_ONNX_PATH,
    imgsz: int = DEFAULT_IMAGE_SIZE,
) -> Path:
    """
    One-time conversion of the .pt weights to ONNX. Only this function
    imports ultralytics/torch; the runtime hot path never does.
    """
    if onnx_path.exists():
        return onnx_path
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Neither {onnx_path} nor {pt_path} exists. Run scripts/fetch_models.py first."
        )
    logger.info("Exporting %s -> ONNX (one time, ~20s)...", pt_path.name)
    from ultralytics import YOLO

    exported = YOLO(str(pt_path)).export(
        format="onnx", imgsz=imgsz, opset=13, simplify=False, dynamic=False
    )
    Path(exported).replace(onnx_path)
    return onnx_path


class ObjectDetector:
    """
    Given an image, return real detections above a confidence threshold,
    restricted to classes relevant to an assistive agent.

    Deliberately does NOT do tracking, direction, or distance — see
    ObjectTracker and the direction/distance modules (Segments 3, 5, 6).
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        image_size: int = DEFAULT_IMAGE_SIZE,
        backend: Optional[InferenceBackend] = None,
        relevant_classes: Optional[set] = None,
    ) -> None:
        onnx_path = Path(weights_path) if weights_path else ensure_onnx_export()
        self.backend = backend or select_backend(onnx_path)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.relevant_classes = (
            RELEVANT_CLASSES if relevant_classes is None else relevant_classes
        )
        self.last_inference_seconds: Optional[float] = None
        logger.info("ObjectDetector using backend=%s", self.backend.name)

    def detect(self, image: Union[str, Path, np.ndarray]) -> List[Detection]:
        """
        `image` may be a file path or a BGR numpy array (what OpenCV and the
        WebSocket frame decoder both produce).
        """
        if isinstance(image, (str, Path)):
            frame = cv2.imread(str(image))
            if frame is None:
                raise FileNotFoundError(f"could not read image: {image}")
        else:
            frame = image

        start = time.perf_counter()
        batch, info = preprocess(frame, self.image_size)
        raw = self.backend.infer(batch)
        results = postprocess(
            raw, info, frame.shape[:2], conf_threshold=self.confidence_threshold
        )
        self.last_inference_seconds = time.perf_counter() - start

        height, width = frame.shape[:2]
        now = time.time()
        return [
            Detection(
                class_name=r.class_name,
                confidence=r.confidence,
                box=BoundingBox(*r.xyxy),
                timestamp=now,
                image_width=width,
                image_height=height,
            )
            for r in results
            if r.class_name in self.relevant_classes
        ]
