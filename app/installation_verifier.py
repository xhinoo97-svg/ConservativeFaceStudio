from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.model_catalog import all_model_manifests
from app.model_registry import inspect_model
from app.model_runtime_registry import ACTIVE, FALLBACK
from app.opencv_lama import OpenCVLamaEngine
from app.opencv_nafnet import NafNetDeblurEngine
from app.opencv_semantic_models import FaceParsingEngine, HeadPoseEngine
from app.paths import runtime_root, user_data_root


def _production_manifests():
    by_key = {item.key: item for item in all_model_manifests()}
    missing = sorted((ACTIVE | FALLBACK) - set(by_key))
    if missing:
        raise RuntimeError(f"Production manifest missing: {missing}")
    return {key: by_key[key] for key in sorted(ACTIVE | FALLBACK)}


def verify_installation(root: str | Path | None = None) -> dict[str, Any]:
    """Verify packaged production files without downloading or contacting the network."""
    base = Path(root).resolve() if root is not None else runtime_root().resolve()
    models: dict[str, Any] = {}
    failures: list[str] = []
    for key, manifest in _production_manifests().items():
        try:
            status = inspect_model(manifest, base)
        except Exception as exc:
            status = {"exists": False, "checksum_ok": False, "error": str(exc)}
        exists = bool(status.get("exists", False))
        checksum_ok = status.get("checksum_ok") is True
        models[key] = {
            "path": status.get("path"),
            "exists": exists,
            "checksum_ok": checksum_ok,
            "size_bytes": status.get("size_bytes"),
        }
        if not exists:
            failures.append(f"missing:{key}")
        elif not checksum_ok:
            failures.append(f"checksum:{key}")

    writable_root = user_data_root()
    writable_ok = False
    try:
        writable_root.mkdir(parents=True, exist_ok=True)
        probe = writable_root / ".write-probe"
        probe.write_bytes(b"ok")
        probe.unlink(missing_ok=True)
        writable_ok = True
    except OSError as exc:
        failures.append(f"user_data_not_writable:{exc}")

    backend = {
        "FaceDetectorYN": hasattr(cv2, "FaceDetectorYN"),
        "FaceRecognizerSF": hasattr(cv2, "FaceRecognizerSF"),
        "OpenCL_available": bool(getattr(cv2, "ocl", None) and cv2.ocl.haveOpenCL()),
    }
    if not backend["FaceDetectorYN"]:
        failures.append("backend:FaceDetectorYN")
    if not backend["FaceRecognizerSF"]:
        failures.append("backend:FaceRecognizerSF")

    return {
        "ok": not failures,
        "offline": True,
        "network_used": False,
        "runtime_root": str(base),
        "user_data_root": str(writable_root),
        "user_data_writable": writable_ok,
        "models": models,
        "backend": backend,
        "failures": failures,
    }


def _synthetic_face(size: int = 128) -> np.ndarray:
    image = np.full((size, size, 3), 36, np.uint8)
    c = size // 2
    cv2.ellipse(image, (c, c + 2), (size // 4, size // 3), 0, 0, 360, (155, 178, 205), -1)
    cv2.circle(image, (c - 13, c - 10), 4, (28, 28, 28), -1)
    cv2.circle(image, (c + 13, c - 10), 4, (28, 28, 28), -1)
    cv2.line(image, (c, c - 3), (c, c + 15), (80, 90, 100), 2)
    cv2.line(image, (c - 12, c + 27), (c + 12, c + 27), (55, 55, 70), 2)
    return image


def offline_inference_test(root: str | Path | None = None) -> dict[str, Any]:
    """Run real CPU inference from packaged weights only; never invokes bootstrap/download."""
    report = verify_installation(root)
    if not report["ok"]:
        return {**report, "inference_ok": False}

    base = Path(report["runtime_root"])
    image = _synthetic_face(128)
    checks: dict[str, Any] = {}
    try:
        yunet = base / "models/opencv_zoo/face_detection_yunet_2023mar.onnx"
        sface = base / "models/opencv_zoo/face_recognition_sface_2021dec.onnx"
        detector = cv2.FaceDetectorYN.create(
            str(yunet), "", (128, 128), 0.1, 0.3, 5000,
            cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
        )
        detector.setInputSize((128, 128))
        _status, faces = detector.detect(image)
        if faces is not None and not np.isfinite(np.asarray(faces, np.float32)).all():
            raise RuntimeError("YuNet non-finite output")
        recognizer = cv2.FaceRecognizerSF.create(
            str(sface), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
        )
        feature = np.asarray(recognizer.feature(cv2.resize(image, (112, 112))), np.float32).reshape(-1)
        if feature.size == 0 or not np.isfinite(feature).all():
            raise RuntimeError("SFace invalid embedding")
        checks["face"] = {"ok": True, "embedding_dim": int(feature.size)}

        nafnet = NafNetDeblurEngine(
            base / "models/nafnet/deblurring_nafnet_2025may.onnx",
            target="cpu", tile_size=128, overlap=16,
        ).infer(image)
        if nafnet.shape != image.shape or not np.isfinite(nafnet).all():
            raise RuntimeError("NAFNet invalid output")
        checks["nafnet"] = {"ok": True, "shape": list(nafnet.shape)}

        labels = FaceParsingEngine(
            base / "models/face_parsing/resnet18.onnx", target="cpu"
        ).predict(image)
        if labels.shape != image.shape[:2] or not np.isfinite(labels).all():
            raise RuntimeError("Face parsing invalid output")
        checks["parsing"] = {"ok": True, "shape": list(labels.shape)}

        pose = HeadPoseEngine(
            base / "models/head_pose/mobilenetv2.onnx", target="cpu"
        ).estimate(image)
        if len(pose) != 3 or not all(np.isfinite(value) for value in pose):
            raise RuntimeError("Head-pose invalid output")
        checks["headpose"] = {"ok": True, "degrees": [float(v) for v in pose]}

        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(mask, (54, 54), (74, 74), 255, -1)
        lama = OpenCVLamaEngine(
            base / "models/lama/inpainting_lama_2025jan.onnx", target="cpu", cpu_threads=2
        ).infer(image, mask)
        if lama.image.shape != image.shape or not np.isfinite(lama.image).all():
            raise RuntimeError("LaMa invalid output")
        if not np.array_equal(lama.image[mask == 0], image[mask == 0]):
            raise RuntimeError("LaMa changed pixels outside requested residual")
        checks["lama"] = {"ok": True, "generated_pixels": int(lama.generated_pixels)}
    except Exception as exc:
        return {**report, "inference_ok": False, "inference_error": str(exc), "checks": checks}

    return {**report, "inference_ok": True, "checks": checks}


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
