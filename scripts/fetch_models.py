"""
Download the model weights Vidi needs at runtime.

The weights are ~163 MB and are deliberately NOT in git (see .gitignore), so
this runs once after cloning and again as part of any deployment build.

WHY THIS MATTERS FOR DEPLOYMENT

The perception hot path reads ONNX directly through OpenVINO or ONNX Runtime.
torch and ultralytics are needed ONLY to convert yolov8n.pt into ONNX the first
time - roughly a gigabyte of dependencies for a one-off export. So the YOLO
ONNX file is committed to the repo, and this script fetches the rest. A
deployment therefore needs neither torch nor ultralytics.

Every download is verified by size before being kept: a truncated model file
fails at load time with an unhelpful error, and half a model is worse than
none.

Usage:  python scripts/fetch_models.py [--force]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "models" / "weights"

# name -> (url, approximate bytes, what it is)
MODELS = {
    "depth_anything_v2_small.onnx": (
        "https://huggingface.co/onnx-community/depth-anything-v2-small/"
        "resolve/main/onnx/model.onnx",
        99_060_839,
        "monocular depth - distance ranges",
    ),
    "face_detection_yunet.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        232_589,
        "face detection - finds faces, does not identify them",
    ),
    "face_recognition_sface.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        38_696_353,
        "face embeddings - only ever matched against people the user enrolled",
    ),
    "mobilenetv2_embed.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/"
        "classification/mobilenet/model/mobilenetv2-12.onnx",
        13_964_571,
        "appearance embeddings - telling the user's backpack from another",
    ),
}

# Tolerance on the expected size; upstream files change slightly over time.
SIZE_TOLERANCE = 0.10


def download(name: str, url: str, expected: int, force: bool) -> bool:
    target = WEIGHTS_DIR / name

    if target.exists() and not force:
        actual = target.stat().st_size
        if abs(actual - expected) <= expected * SIZE_TOLERANCE:
            print("  %-34s present  (%.1f MB)" % (name, actual / 1e6))
            return True
        print("  %-34s WRONG SIZE (%.1f MB, expected ~%.1f MB) - refetching"
              % (name, actual / 1e6, expected / 1e6))

    print("  %-34s downloading %.1f MB..." % (name, expected / 1e6))
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=180) as response, \
                open(temporary, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    except Exception as exc:  # noqa: BLE001
        temporary.unlink(missing_ok=True)
        print("  %-34s FAILED: %s" % (name, str(exc)[:80]))
        return False

    actual = temporary.stat().st_size
    if abs(actual - expected) > expected * SIZE_TOLERANCE:
        # A truncated model fails much later with a confusing error, so refuse
        # it here where the cause is obvious.
        temporary.unlink(missing_ok=True)
        print("  %-34s REJECTED: got %.1f MB, expected ~%.1f MB"
              % (name, actual / 1e6, expected / 1e6))
        return False

    temporary.replace(target)
    print("  %-34s done     (%.1f MB)" % (name, actual / 1e6))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Vidi's model weights.")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file already exists")
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching model weights into %s\n" % WEIGHTS_DIR)

    ok = True
    for name, (url, size, purpose) in MODELS.items():
        print("  %s" % purpose)
        ok &= download(name, url, size, args.force)
        print()

    # yolov8n.onnx ships in the repo precisely so deployments do not need
    # torch; warn loudly rather than silently falling back to a 1 GB install.
    yolo = WEIGHTS_DIR / "yolov8n.onnx"
    if yolo.exists():
        print("  %-34s present  (%.1f MB, committed)"
              % ("yolov8n.onnx", yolo.stat().st_size / 1e6))
    else:
        print("  yolov8n.onnx MISSING. It is committed to the repo; if it is "
              "gone,\n  regenerate it with ultralytics:\n"
              "      python -c \"from ultralytics import YOLO; "
              "YOLO('models/weights/yolov8n.pt')"
              ".export(format='onnx', imgsz=480, opset=13)\"")
        ok = False

    print("\n" + ("All models ready." if ok else "Some models are missing - see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
