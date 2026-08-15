from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import app  # noqa: F401  # Apply the same boot policy used by the packaged executable.
import cv2
import numpy as np

from app.opencv_nafnet import NafNetDeblurEngine
from app.opencv_semantic_models import FaceParsingEngine, HeadPoseEngine


MODEL_CHOICES = ("all", "face", "nafnet", "parsing", "headpose")


def _synthetic_face(size: int = 128) -> np.ndarray:
    image = np.full((size, size, 3), 32, dtype=np.uint8)
    center = size // 2
    cv2.ellipse(image, (center, center + 3), (size // 4, size // 3), 0, 0, 360, (150, 175, 200), -1)
    cv2.circle(image, (center - size // 10, center - size // 12), max(2, size // 32), (25, 25, 25), -1)
    cv2.circle(image, (center + size // 10, center - size // 12), max(2, size // 32), (25, 25, 25), -1)
    cv2.line(image, (center, center - size // 20), (center, center + size // 10), (80, 90, 105), 2)
    cv2.line(image, (center - size // 10, center + size // 5), (center + size // 10, center + size // 5), (55, 55, 70), 2)
    return image


def _require(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"Required pretrained model missing: {path}")
    return path


def _smoke_face(root: Path, image: np.ndarray) -> dict[str, object]:
    yunet = _require(root / "models/detection/face_detection_yunet_2023mar.onnx")
    sface = _require(root / "models/identity/face_recognition_sface_2021dec.onnx")
    if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
        raise SystemExit("[smoke:face] OpenCV build lacks YuNet/SFace APIs")

    print(f"[smoke:face] loading YuNet {yunet}", flush=True)
    detector = cv2.FaceDetectorYN.create(
        str(yunet), "", (image.shape[1], image.shape[0]), 0.1, 0.3, 5000,
        cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
    )
    detector.setInputSize((image.shape[1], image.shape[0]))
    try:
        _status, faces = detector.detect(image)
    except cv2.error as exc:
        raise SystemExit(f"[smoke:face] YuNet CPU inference failed: {exc}") from exc
    face_count = 0 if faces is None else int(len(faces))
    if faces is not None and not np.isfinite(np.asarray(faces, dtype=np.float32)).all():
        raise SystemExit("[smoke:face] YuNet returned non-finite detection values")

    print(f"[smoke:face] loading SFace {sface}", flush=True)
    recognizer = cv2.FaceRecognizerSF.create(
        str(sface), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
    )
    # SFace feature() consumes an aligned 112x112 face. For a smoke test we only need
    # a real forward pass through the checkpoint; detector accuracy is tested elsewhere.
    aligned = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
    try:
        feature = recognizer.feature(aligned)
    except cv2.error as exc:
        raise SystemExit(f"[smoke:face] SFace CPU inference failed: {exc}") from exc
    embedding = np.asarray(feature, dtype=np.float32).reshape(-1)
    if embedding.size == 0 or not np.isfinite(embedding).all():
        raise SystemExit("[smoke:face] SFace returned an invalid embedding")

    result = {"yunet_faces": face_count, "sface_embedding_dim": int(embedding.size)}
    print(f"[smoke:face] PASS {result}", flush=True)
    return result


def _smoke_nafnet(root: Path, image: np.ndarray) -> dict[str, object]:
    path = _require(root / "models/deblur/deblurring_nafnet_2025may.onnx")
    print(f"[smoke:nafnet] loading {path}", flush=True)
    restored = NafNetDeblurEngine(path, target="cpu", tile_size=128, overlap=16).infer(image)
    if restored.shape != image.shape or restored.dtype != np.uint8 or not np.isfinite(restored).all():
        raise SystemExit("[smoke:nafnet] CPU inference returned an invalid image")
    result = {"shape": list(restored.shape), "dtype": str(restored.dtype)}
    print(f"[smoke:nafnet] PASS {result}", flush=True)
    return result


def _smoke_parsing(root: Path, image: np.ndarray) -> dict[str, object]:
    path = _require(root / "models/parsing/resnet18.onnx")
    print(f"[smoke:parsing] loading {path}", flush=True)
    labels = FaceParsingEngine(path, target="cpu").predict(image)
    if labels.shape != image.shape[:2] or labels.dtype != np.uint8:
        raise SystemExit("[smoke:parsing] CPU inference returned an invalid label map")
    max_class = int(labels.max(initial=0))
    if max_class > 18:
        raise SystemExit(f"[smoke:parsing] class id outside CelebAMask-HQ range: {max_class}")
    result = {
        "shape": list(labels.shape),
        "classes_seen": sorted(int(value) for value in np.unique(labels)),
    }
    print(f"[smoke:parsing] PASS {result}", flush=True)
    return result


def _smoke_headpose(root: Path, image: np.ndarray) -> dict[str, object]:
    path = _require(root / "models/pose/mobilenetv2.onnx")
    print(f"[smoke:headpose] loading {path}", flush=True)
    pose = HeadPoseEngine(path, target="cpu").estimate(image)
    if len(pose) != 3 or not all(np.isfinite(value) for value in pose):
        raise SystemExit(f"[smoke:headpose] CPU inference returned invalid angles: {pose}")
    if any(abs(float(value)) > 180.0 for value in pose):
        raise SystemExit(f"[smoke:headpose] CPU inference returned implausible angles: {pose}")
    result = {"degrees": [float(value) for value in pose]}
    print(f"[smoke:headpose] PASS {result}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real pretrained-model CPU smoke checks")
    parser.add_argument("--only", choices=MODEL_CHOICES, default="all")
    args = parser.parse_args()

    root = Path(".").resolve()
    image = _synthetic_face(128)
    requested = MODEL_CHOICES[1:] if args.only == "all" else (args.only,)
    checks = {
        "face": _smoke_face,
        "nafnet": _smoke_nafnet,
        "parsing": _smoke_parsing,
        "headpose": _smoke_headpose,
    }

    report: dict[str, object] = {"backend": "cpu"}
    for name in requested:
        report[name] = checks[name](root, image)
    print(report, flush=True)


if __name__ == "__main__":
    main()
