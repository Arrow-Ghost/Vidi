"""
Inference runtime abstraction — the seam between "what model" and "what
hardware actually runs it".

WHY THIS EXISTS (measured, not assumed — see docs/PERFORMANCE.md):
On this machine a single YOLOv8n forward pass costs, for the same weights
and the same 480x480 input:

    torch 2.13.0+cpu        3614.5 ms     0.3 FPS
    OpenVINO CPU             498.7 ms     2.0 FPS
    ONNX Runtime CPU (4t)    126.6 ms     7.9 FPS
    ONNX Runtime CPU (2t)     55.3 ms    18.1 FPS
    OpenVINO Intel iGPU        8.9 ms   112.0 FPS   <-- default

Torch's CPU convolution path is pathologically slow here despite MKL-DNN
being present and enabled, so it is never used for inference. The iGPU is
preferred not only because it is fastest but because this laptop's CPU sits
near 100% utilization from other software — moving perception onto the
otherwise-idle Iris Xe is what makes the rest of the system responsive.
Note ONNX Runtime is FASTER with 2 threads than 4 for the same reason.

Every backend here consumes the identical ONNX file and returns the identical
raw tensor, so switching hardware can never change detection semantics.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger("sfcc.perception.runtime")

# Measured sweet spot on a contended 4-core machine; see table above.
ORT_INTRA_OP_THREADS = 2


class InferenceBackend(ABC):
    """Runs a preprocessed NCHW float32 batch and returns raw model output."""

    name: str = "abstract"

    @abstractmethod
    def infer(self, batch: np.ndarray) -> np.ndarray: ...

    @property
    @abstractmethod
    def available(self) -> bool: ...


class OpenVinoBackend(InferenceBackend):
    """Intel iGPU (or CPU) via OpenVINO. Reads ONNX directly — no separate export step."""

    def __init__(self, model_path: Path, device: str = "GPU") -> None:
        self.name = f"openvino:{device}"
        self._compiled = None
        self._device = device
        try:
            import openvino as ov

            core = ov.Core()
            if device not in core.available_devices:
                raise RuntimeError(f"device {device} not in {core.available_devices}")
            model = core.read_model(str(model_path))
            # LATENCY hint optimizes for single-stream response time, which is
            # what a real-time assistant needs — not batch throughput.
            self._compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
            logger.info("OpenVINO backend ready on %s", device)
        except Exception as exc:
            logger.warning("OpenVINO backend unavailable on %s: %s", device, exc)
            self._compiled = None

    @property
    def available(self) -> bool:
        return self._compiled is not None

    def infer(self, batch: np.ndarray) -> np.ndarray:
        result = self._compiled(batch)
        return np.asarray(result[self._compiled.output(0)])


class OnnxRuntimeBackend(InferenceBackend):
    """Portable CPU fallback. Works on any machine with no GPU at all."""

    def __init__(self, model_path: Path, threads: int = ORT_INTRA_OP_THREADS) -> None:
        self.name = f"onnxruntime:cpu:{threads}t"
        self._session = None
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(model_path), options, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            logger.info("ONNX Runtime backend ready (%d threads)", threads)
        except Exception as exc:
            logger.warning("ONNX Runtime backend unavailable: %s", exc)
            self._session = None

    @property
    def available(self) -> bool:
        return self._session is not None

    def infer(self, batch: np.ndarray) -> np.ndarray:
        return np.asarray(self._session.run(None, {self._input_name: batch})[0])


def select_backend(
    model_path: Path, preferred: Optional[List[str]] = None
) -> InferenceBackend:
    """
    Pick the fastest backend that actually loads on this machine, in
    measured-performance order. Raises if none work rather than silently
    degrading to something that cannot meet real-time requirements —
    a perception system that quietly runs at 0.3 FPS is worse than one
    that refuses to start and says why.
    """
    order = preferred or ["openvino:GPU", "onnxruntime:cpu", "openvino:CPU"]
    attempted: List[str] = []
    for spec in order:
        if spec.startswith("openvino:"):
            backend: InferenceBackend = OpenVinoBackend(model_path, spec.split(":", 1)[1])
        elif spec.startswith("onnxruntime"):
            backend = OnnxRuntimeBackend(model_path)
        else:
            continue
        attempted.append(spec)
        if backend.available:
            return backend
    raise RuntimeError(
        f"No usable inference backend. Tried: {attempted}. "
        "Install onnxruntime (CPU fallback) or openvino (Intel GPU)."
    )


def benchmark(backend: InferenceBackend, shape=(1, 3, 480, 480), iterations: int = 20) -> dict:
    """Real measured latency for this backend — used by /health and docs."""
    x = np.random.rand(*shape).astype(np.float32)
    for _ in range(5):
        backend.infer(x)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        backend.infer(x)
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    p50 = times[len(times) // 2]
    return {
        "backend": backend.name,
        "p50_ms": round(p50, 2),
        "p95_ms": round(times[int(len(times) * 0.95) - 1], 2),
        "max_fps": round(1000 / p50, 1),
    }
