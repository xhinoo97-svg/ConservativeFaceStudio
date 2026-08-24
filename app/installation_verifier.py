from __future__ import annotations

import json
import socket
import tempfile
import urllib.request
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
    required_directories = (
        "models/detection", "models/identity", "models/landmarks", "models/parsing",
        "models/pose", "models/deblur", "models/reference", "models/inpainting",
        "models/restoration", "models/optional", "config", "licenses", "runtime", "logs",
        "projects", "exports", "cache",
    )
    directories = {name: (base / name).is_dir() for name in required_directories}
    failures.extend(f"missing_directory:{name}" for name, present in directories.items() if not present)
    metadata: dict[str, Any] = {}
    production_keys = set(_production_manifests())
    for name, expected_container in (
        ("models/model-registry.json", dict),
        ("models/model-manifests.json", list),
    ):
        path = base / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, expected_container):
                raise ValueError(f"expected {expected_container.__name__}")
            if isinstance(payload, dict):
                raw_entries = payload.get("models", [])
            else:
                raw_entries = payload
            if not isinstance(raw_entries, list):
                raise ValueError("models must be a list")
            registered = {
                str(item.get("key"))
                for item in raw_entries
                if isinstance(item, dict) and item.get("key")
            }
            missing_keys = sorted(production_keys - registered)
            if missing_keys:
                raise ValueError(f"production keys missing: {missing_keys}")
            metadata[name] = {"ok": True, "model_count": len(registered)}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            metadata[name] = {"ok": False, "error": str(exc)}
            failures.append(f"metadata:{name}")
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
        "metadata": metadata,
        "directories": directories,
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
    network_attempts: list[str] = []
    original_urlopen = urllib.request.urlopen
    original_create_connection = socket.create_connection
    original_socket_connect = socket.socket.connect

    def blocked_network(*args, **kwargs):
        network_attempts.append(str(args[0]) if args else "unknown")
        raise OSError("Network disabled by Conservative Face Studio offline test")

    urllib.request.urlopen = blocked_network
    socket.create_connection = blocked_network
    socket.socket.connect = blocked_network
    try:
        yunet = base / "models/detection/face_detection_yunet_2023mar.onnx"
        sface = base / "models/identity/face_recognition_sface_2021dec.onnx"
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
            base / "models/deblur/deblurring_nafnet_2025may.onnx",
            target="cpu", tile_size=128, overlap=16,
        ).infer(image)
        if nafnet.shape != image.shape or not np.isfinite(nafnet).all():
            raise RuntimeError("NAFNet invalid output")
        checks["nafnet"] = {"ok": True, "shape": list(nafnet.shape)}

        labels = FaceParsingEngine(
            base / "models/parsing/resnet18.onnx", target="cpu"
        ).predict(image)
        if labels.shape != image.shape[:2] or not np.isfinite(labels).all():
            raise RuntimeError("Face parsing invalid output")
        checks["parsing"] = {"ok": True, "shape": list(labels.shape)}

        pose = HeadPoseEngine(
            base / "models/pose/mobilenetv2.onnx", target="cpu"
        ).estimate(image)
        if len(pose) != 3 or not all(np.isfinite(value) for value in pose):
            raise RuntimeError("Head-pose invalid output")
        checks["headpose"] = {"ok": True, "degrees": [float(v) for v in pose]}

        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(mask, (54, 54), (74, 74), 255, -1)
        lama = OpenCVLamaEngine(
            base / "models/inpainting/inpainting_lama_2025jan.onnx", target="cpu", cpu_threads=2
        ).infer(image, mask)
        if lama.image.shape != image.shape or not np.isfinite(lama.image).all():
            raise RuntimeError("LaMa invalid output")
        if not np.array_equal(lama.image[mask == 0], image[mask == 0]):
            raise RuntimeError("LaMa changed pixels outside requested residual")
        checks["lama"] = {"ok": True, "generated_pixels": int(lama.generated_pixels)}

        from app.automatic import AutomaticPipelineRunner
        from app.execution import Workspace
        from app.production_models import resolve_local_production_models

        local_pack = resolve_local_production_models(base, user_data_root())
        if not (local_pack.face_ready and local_pack.standard_ready and local_pack.inpaint_ready):
            raise RuntimeError(f"Offline production router incomplete: {local_pack.errors}")
        model_paths = {key: str(path) for key, path in local_pack.paths.items()}
        main = image.copy()
        cv2.rectangle(main, (40, 43), (88, 72), (12, 12, 12), -1)
        workspace = Workspace(
            primary=main,
            # Exercise plural-reference import/routing in the installed package.  The
            # dedicated release smoke covers the complete 0..9 matrix; this compact
            # path proves the packaged executable can consume multiple references
            # and export while the network is physically blocked.
            references=[image.copy(), cv2.flip(image, 1)],
            metadata={
                "core_model_paths": model_paths,
                "user_selected_primary": True,
                "primary_priority_policy": "fixed-photo-1-main-image",
            },
        )
        with tempfile.TemporaryDirectory(prefix="cfs-offline-pipeline-") as directory:
            result = AutomaticPipelineRunner(workspace).run(Path(directory) / "final.png", upscale=1)
            final = cv2.imread(str(result.final_image), cv2.IMREAD_COLOR)
            if final is None or final.shape != main.shape:
                raise RuntimeError("Offline representative pipeline produced an invalid output")
            if workspace.provenance_map is None or workspace.provenance_map.shape != main.shape[:2]:
                raise RuntimeError("Offline representative pipeline produced invalid provenance")
        if network_attempts:
            raise RuntimeError(f"Offline pipeline attempted network access: {network_attempts}")
        checks["representative_pipeline"] = {
            "ok": True,
            "references": 2,
            "main_shape_preserved": True,
            "provenance_valid": True,
        }
    except Exception as exc:
        return {**report, "inference_ok": False, "inference_error": str(exc), "checks": checks}
    finally:
        urllib.request.urlopen = original_urlopen
        socket.create_connection = original_create_connection
        socket.socket.connect = original_socket_connect

    return {**report, "inference_ok": True, "network_attempts": network_attempts, "checks": checks}


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
