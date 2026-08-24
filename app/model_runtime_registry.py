from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core_model_fallbacks import exportable_core_fallbacks
from app.model_catalog import all_model_manifests
from app.model_registry import inspect_model
from app.optional_heavy_models import heavy_profile_by_key


# Release activation is explicit. A manifest being conservative_default does not prove
# that its loader, inference routing, smoke test and packaged weights are production-ready.
ACTIVE = {
    "opencv_yunet",
    "opencv_sface",
    "opencv_nafnet_deblur",
    "face_parsing_resnet18_onnx",
    "head_pose_mobilenetv2_onnx",
}
FALLBACK = {"opencv_lama_inpaint"}
TESTING = {
    "mediapipe_face_landmarker",
    "bisenet_face_parsing",
    "3ddfa_mb1",
    "dmdnet",
    "restormer_motion_deblur",
    "restormer_real_denoise",
    "realesrgan_x2plus",
    "codeformer_v010",
    "gfpgan_v13",
    "restoreformer_v13_asset",
}
DISABLED = {"insightface_identity"}
OPTIONAL_RESEARCH = {"refstar_research", "instantrestore_research", "osdface_research"}


def _declared_status(key: str) -> str:
    if key in ACTIVE:
        return "ACTIVE"
    if key in FALLBACK:
        return "FALLBACK"
    if key in DISABLED:
        return "DISABLED"
    if key in OPTIONAL_RESEARCH:
        return "OPTIONAL_RESEARCH"
    # Fail closed: unlisted manifests are candidates, not production backends.
    return "TESTING"


def _function_for(key: str) -> str:
    value = key.lower()
    if "yunet" in value:
        return "face_detection_landmarks"
    if "sface" in value or "insightface" in value:
        return "identity_guardrail"
    if "nafnet" in value or "restormer" in value:
        return "deblur_denoise"
    if "parsing" in value or "bisenet" in value:
        return "face_segmentation"
    if "pose" in value or "3ddfa" in value:
        return "pose_alignment"
    if "lama" in value or "codeformer" in value or "gfpgan" in value or "restoreformer" in value:
        return "inpainting_face_restoration"
    if "dmdnet" in value:
        return "identity_preserving_face_restoration"
    if "esrgan" in value:
        return "super_resolution"
    if "landmarker" in value:
        return "dense_landmarks"
    return "optional_model"


def _backend_for(key: str, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".onnx":
        return "OpenCV-DNN/ONNXRuntime"
    if suffix in {".pth", ".pt"}:
        return "PyTorch adapter required"
    if suffix == ".task":
        return "MediaPipe Tasks"
    return "model-specific"


def _tier_for(status: str, key: str) -> str:
    if status in {"ACTIVE", "FALLBACK"}:
        return "TIER_A_CORE_PRODUCTION"
    if key in {"codeformer_v010", "gfpgan_v13", "restoreformer_v13_asset", "dmdnet"}:
        return "TIER_C_HEAVY_OPTIONAL"
    if status in {"DISABLED", "OPTIONAL_RESEARCH"}:
        return "TIER_D_RESEARCH_DISABLED"
    return "TIER_D_RESEARCH_UNSUPPORTED"


def _capabilities(key: str, filename: str, status: str) -> dict[str, object]:
    suffix = Path(filename).suffix.lower()
    onnx = suffix == ".onnx"
    production = status in {"ACTIVE", "FALLBACK"}
    return {
        "cpu_support": bool(onnx or production),
        "opencl_support": key in {"opencv_yunet", "opencv_sface"},
        "cuda_requirement": bool(suffix in {".pth", ".pt"} and not production),
        "onnx_support": onnx,
        "vulkan_support": key == "realesrgan_x2plus",
        "expected_runtime": (
            "CPU smoke measured in release workflow; host-dependent"
            if production
            else "Not measured on target hardware; disabled until benchmarked"
        ),
    }


def build_runtime_registry(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root).resolve()
    heavy = heavy_profile_by_key()
    entries: list[dict[str, Any]] = []
    for manifest in all_model_manifests():
        try:
            local = inspect_model(manifest, root_path)
        except Exception as exc:
            local = {"exists": False, "error": str(exc)}
        profile = heavy.get(manifest.key)
        declared = _declared_status(manifest.key)
        checksum_ok = local.get("checksum_ok")
        status = "BROKEN" if bool(local.get("exists", False)) and checksum_ok is False else declared
        stable_active = status in {"ACTIVE", "FALLBACK"}
        capabilities = _capabilities(manifest.key, manifest.filename, status)
        entries.append({
            "key": manifest.key,
            "name": manifest.title,
            "task": _function_for(manifest.key),
            "function": _function_for(manifest.key),
            "version": Path(manifest.filename).stem,
            "path": manifest.destination,
            "sha256_expected": manifest.expected_sha256,
            "sha256_local": local.get("sha256"),
            "size_bytes_local": local.get("size_bytes"),
            "maximum_download_bytes": manifest.max_bytes,
            "model_size_bytes": local.get("size_bytes"),
            "backend": _backend_for(manifest.key, manifest.filename),
            "framework": _backend_for(manifest.key, manifest.filename),
            "ram_estimate_mb": None if profile is None else profile.measured_peak_ram_mb,
            "vram_estimate_mb": None,
            "cpu_gpu": "CPU first; OpenCL only after self-test",
            **capabilities,
            "status": status,
            "production_status": status,
            "installation_tier": _tier_for(status, manifest.key),
            "declared_release_status": declared,
            "stable_active": stable_active,
            "installed": bool(local.get("exists", False)),
            "checksum_ok": checksum_ok,
            "benchmark": (
                "production smoke required"
                if profile is None
                else profile.benchmark_status
            ),
            "conservative_default": manifest.conservative_default,
            "code_license": manifest.code_license,
            "weights_license": manifest.weights_license,
            "source_url": manifest.source_url,
            "notes": manifest.notes,
        })
    return {
        "format": "ConservativeFaceStudio model runtime registry",
        "version": 2,
        "hardware_target": "runtime-detected; no CPU, RAM, GPU or driver is assumed",
        "states": ["ACTIVE", "TESTING", "FALLBACK", "DISABLED", "OPTIONAL_RESEARCH", "BROKEN"],
        "activation_policy": "explicit-only; manifest presence/conservative_default never auto-promotes a backend",
        "models": entries,
        "core_fallback_chains": exportable_core_fallbacks(),
    }


def export_runtime_registry(path: str | Path, root: str | Path = ".") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_runtime_registry(root), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target
