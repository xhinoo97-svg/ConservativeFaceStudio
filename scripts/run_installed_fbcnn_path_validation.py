from __future__ import annotations

"""Exercise real LR-ASPP -> JPEG router -> FBCNN through PipelineWorker.

This is a DEVELOPMENT installed-path validation, not a quality benchmark and not a
production qualification. It deliberately uses no reference image and requires the
normal local model-pack resolvers, feature flag, worker, automatic runner and Block 8
bridge to be active. Non-production FBCNN pixels must remain shadow evidence only.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2
import numpy as np

from app.execution import Workspace
from app.fbcnn_upstream_backend import (
    APPROVED_CHECKPOINT_SHA256,
    CONSERVATIVE_RESTORATION_FRACTION,
    OFFICIAL_REPOSITORY,
    PINNED_REVISION,
)
from app.paper_quality_model_pack import (
    LRASPP_ONNX_SHA256,
    inspect_paper_quality_validation_pack,
)
from app.production_models import resolve_local_production_models
from app.settings import load_runtime_settings
from app.worker import PipelineWorker


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_image(image: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(image).tobytes())


def _letterbox_portrait(image: np.ndarray, size: int = 384) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Public DEVELOPMENT portrait must be a uint8 image")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 64:
        raise ValueError("Public DEVELOPMENT portrait has invalid geometry")
    target = int(size)
    if target < 192 or target % 8:
        raise ValueError("Installed-path validation canvas must be >=192 and divisible by 8")
    height, width = image.shape[:2]
    scale = min(float(target) / height, float(target) / width)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    top = (target - resized.shape[0]) // 2
    left = (target - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def _jpeg_round_trip(image: np.ndarray, quality: int = 10) -> tuple[np.ndarray, bytes]:
    qf = int(quality)
    if not 1 <= qf <= 20:
        raise ValueError("Installed-path JPEG stress quality must be in [1, 20]")
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), qf],
    )
    if not ok:
        raise RuntimeError("Could not encode DEVELOPMENT JPEG input")
    payload = encoded.tobytes()
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != image.shape:
        raise RuntimeError("Could not decode DEVELOPMENT JPEG input")
    return decoded, payload


def _validation_candidate(details: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = details.get("validation_model_candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise RuntimeError("Installed path must report exactly one validation candidate")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise RuntimeError("Installed FBCNN candidate report is malformed")
    return candidate


def validate_installed_block(
    details: Mapping[str, Any],
    *,
    block_image: np.ndarray,
    immutable_main: np.ndarray,
) -> dict[str, Any]:
    """Validate truthfulness and fail-closed final-pixel authority for Block 8."""
    if details.get("engine") != "installed-paper-quality-runtime-v1":
        raise RuntimeError("Block 8 did not execute InstalledPaperQualityRuntime")
    if details.get("paper_quality_runtime_wired") is not True:
        raise RuntimeError("Paper Quality runtime is not wired")
    damage = details.get("damage")
    if not isinstance(damage, dict) or damage.get("dominant_damage_class") != "JPEG_ARTIFACT":
        raise RuntimeError(f"LR-ASPP did not classify the input as JPEG_ARTIFACT: {damage}")
    route = details.get("damage_route")
    if not isinstance(route, dict) or route.get("damage_kind") not in {"JPEG_ARTIFACT", "MIXED"}:
        raise RuntimeError(f"Damage router did not retain dominant JPEG evidence: {route}")
    model_route = details.get("validation_model_route")
    if (
        not isinstance(model_route, dict)
        or model_route.get("damage_kind") != "JPEG_ARTIFACT"
        or model_route.get("source_damage_class") != "JPEG_ARTIFACT"
    ):
        raise RuntimeError(f"FBCNN was not bound to a JPEG-only model route: {model_route}")
    class_evidence = damage.get("admitted_class_evidence")
    if not isinstance(class_evidence, list) or not any(
        isinstance(item, dict) and item.get("damage_class") == "JPEG_ARTIFACT"
        for item in class_evidence
    ):
        raise RuntimeError(f"Admitted JPEG class evidence is missing: {class_evidence}")
    if route.get("damage_kind") == "MIXED":
        if len(class_evidence) < 2:
            raise RuntimeError("MIXED route does not report multiple admitted damage classes")
        parent_pixels = int(route.get("mask_pixels", 0))
        model_pixels = int(model_route.get("mask_pixels", 0))
        if not 0 < model_pixels < parent_pixels:
            raise RuntimeError(
                "JPEG validation subroute is not a strict subset of the MIXED route: "
                f"model={model_pixels} parent={parent_pixels}"
            )
    executed = details.get("models_actually_executed")
    if not isinstance(executed, list) or len(executed) != 1:
        raise RuntimeError(
            "Installed path did not report exactly one executed model: "
            f"{details.get('model_execution_errors')}"
        )
    model = executed[0]
    if not isinstance(model, dict) or model.get("model_key") != "fbcnn":
        raise RuntimeError(f"Installed path executed the wrong model: {model}")
    if model.get("checkpoint_sha256") != APPROVED_CHECKPOINT_SHA256:
        raise RuntimeError("Installed FBCNN checkpoint identity does not match the approved hash")
    if model.get("execution_scope") != "INSTALLED_PATH_VALIDATION_SHADOW":
        raise RuntimeError("Installed FBCNN execution scope is not validation shadow")
    if model.get("fused_to_final") is not False:
        raise RuntimeError("Non-production FBCNN candidate was granted final-pixel authority")
    candidate = _validation_candidate(details)
    resource = candidate.get("resource")
    if not isinstance(resource, dict) or not isinstance(resource.get("post_unload"), dict):
        raise RuntimeError("FBCNN lifecycle report does not prove the post-unload boundary")
    if details.get("validation_candidates_fused_to_final") is not False:
        raise RuntimeError("Validation candidates were fused to final pixels")
    if int(details.get("generated_pixels", -1)) != 0:
        raise RuntimeError("Validation-only FBCNN produced final generated pixels")
    if int(details.get("wrong_person_final_pixels", -1)) != 0:
        raise RuntimeError("Wrong-person final pixels are not zero")
    if int(details.get("provenance_violations", -1)) != 0:
        raise RuntimeError("Provenance violations are not zero")
    if int(details.get("outside_authority_changed_pixels", -1)) != 0:
        raise RuntimeError("Paper Quality changed pixels outside final authority")
    if not np.array_equal(block_image, immutable_main):
        raise RuntimeError("Shadow validation changed the immutable MAIN at Block 8")
    errors = details.get("model_execution_errors")
    if errors != []:
        raise RuntimeError(f"Installed FBCNN execution reported errors: {errors}")
    trace_items = details.get("paper_quality_trace")
    if not isinstance(trace_items, list):
        raise RuntimeError("Paper Quality trace is missing")
    trace = {
        str(item.get("stage")): str(item.get("status"))
        for item in trace_items
        if isinstance(item, dict)
    }
    expected = {
        "DamageMaskRuntime": "EXECUTED",
        "damage_router": "EXECUTED",
        "model_execution": "VALIDATION_EXECUTED_NOT_FUSED",
        "PaperQualityRuntime": "EXECUTED",
        "provenance": "VERIFIED",
    }
    for stage, status in expected.items():
        if trace.get(stage) != status:
            raise RuntimeError(f"Paper Quality trace mismatch for {stage}: {trace.get(stage)}")
    return {
        "damage_class": damage["dominant_damage_class"],
        "damage_confidence": float(damage["dominant_confidence"]),
        "damage_route": route["damage_kind"],
        "model_route": model_route["damage_kind"],
        "admitted_class_evidence": list(class_evidence),
        "decision": str(details.get("decision")),
        "model": dict(model),
        "candidate": dict(candidate),
        "trace": trace,
        "wrong_person_final_pixels": 0,
        "provenance_violations": 0,
        "healthy_pixels_changed": 0,
        "generated_final_pixels": 0,
    }


def run_validation(
    input_path: Path,
    output_root: Path,
    *,
    source_key: str,
    source_license: str,
    source_page_url: str,
    jpeg_quality: int = 10,
) -> dict[str, Any]:
    settings = load_runtime_settings()
    if settings.paper_quality_enabled is not True:
        raise RuntimeError("paper_quality_enabled must be true for installed-path validation")
    production = resolve_local_production_models()
    if not (production.face_ready and production.standard_ready and production.inpaint_ready):
        raise RuntimeError(f"Installed production pack is incomplete: {production.errors}")
    validation_pack = inspect_paper_quality_validation_pack(REPOSITORY_ROOT)
    if not validation_pack.installed_jpeg_route_ready:
        raise RuntimeError(f"Paper Quality validation pack is incomplete: {validation_pack.errors}")

    source_bytes = input_path.read_bytes()
    source = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"Cannot decode public DEVELOPMENT portrait: {input_path}")
    clean = _letterbox_portrait(source)
    degraded, jpeg_bytes = _jpeg_round_trip(clean, jpeg_quality)
    output_root.mkdir(parents=True, exist_ok=True)
    degraded_path = output_root / "development-input-jpeg.jpg"
    degraded_path.write_bytes(jpeg_bytes)

    workspace = Workspace(
        primary=degraded.copy(),
        references=[],
        metadata={
            "user_selected_primary": True,
            "primary_priority_policy": "fixed-photo-1-main-image",
            "checkpoint_directory": str(output_root / "checkpoints"),
            "validation_scope": "PUBLIC_DEVELOPMENT_INSTALLED_PATH_NO_HOLDOUT",
        },
    )
    worker = PipelineWorker(workspace, output_root / "final.png", upscale=1)
    completed: list[Any] = []
    failures: list[str] = []
    progress: list[dict[str, Any]] = []
    worker.completed.connect(completed.append)
    worker.failed.connect(failures.append)
    worker.progress_detail.connect(lambda value: progress.append(dict(value)))
    worker.run()
    if failures:
        raise RuntimeError(f"PipelineWorker failed: {failures}")
    if len(completed) != 1:
        raise RuntimeError(f"PipelineWorker completion count mismatch: {len(completed)}")
    result = completed[0]
    blocks = [item for item in result.results if item.block == "inpaint"]
    if len(blocks) != 1:
        raise RuntimeError(f"Installed pipeline Block 8 count mismatch: {len(blocks)}")
    block = blocks[0]
    verified = validate_installed_block(
        block.details,
        block_image=block.image,
        immutable_main=degraded,
    )
    block_events = [
        item
        for item in progress
        if item.get("block_index") == 8 and item.get("model_keys") == ["fbcnn"]
    ]
    if not block_events:
        raise RuntimeError("Timeline did not truthfully attribute FBCNN at Block 8")

    report = {
        "schema_version": 1,
        "experiment": "paper_quality_installed_fbcnn_windows_validation_v1",
        "scope": "PUBLIC_DEVELOPMENT_INSTALLED_PATH_NO_HOLDOUT",
        "production_qualified": False,
        "candidate_sha": None,
        "source": {
            "key": source_key,
            "license": source_license,
            "page_url": source_page_url,
            "source_file_sha256": _sha256_bytes(source_bytes),
            "clean_canvas_sha256": _sha256_image(clean),
            "jpeg_input_sha256": _sha256_bytes(jpeg_bytes),
            "jpeg_quality": int(jpeg_quality),
            "references": 0,
            "v3_v4_v5_used": False,
        },
        "model_pack": dict(validation_pack.report),
        "damage_model": {
            "key": "lraspp_damage_mask",
            "onnx_sha256": LRASPP_ONNX_SHA256,
            "production_qualified": False,
        },
        "restoration_model": {
            "key": "fbcnn",
            "official_repository": OFFICIAL_REPOSITORY,
            "official_revision": PINNED_REVISION,
            "checkpoint_sha256": APPROVED_CHECKPOINT_SHA256,
            "conservative_restoration_fraction": CONSERVATIVE_RESTORATION_FRACTION,
            "production_qualified": False,
        },
        "installed_path": {
            "entrypoint": "PipelineWorker.run -> AutomaticPipelineRunner -> InstalledPaperQualityRuntime",
            **verified,
            "timeline_model_attributed": True,
            "final_block_image_sha256": _sha256_image(block.image),
            "immutable_main_sha256": _sha256_image(degraded),
        },
        "gate": "PASS" if verified["healthy_pixels_changed"] == 0 else "FAIL",
        "production_blockers": [
            "validation_candidate_not_fused",
            "final_checkpoint_distribution_terms_pending",
            "offline_installer_not_verified",
            "clean_windows_not_verified",
            "physical_elitebook_not_verified",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-key", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--source-page-url", required=True)
    parser.add_argument("--jpeg-quality", type=int, default=10)
    parser.add_argument("--candidate-sha")
    args = parser.parse_args()
    report = run_validation(
        args.input.resolve(),
        args.output.resolve(),
        source_key=args.source_key,
        source_license=args.source_license,
        source_page_url=args.source_page_url,
        jpeg_quality=args.jpeg_quality,
    )
    report["candidate_sha"] = args.candidate_sha
    destination = args.output / "installed-fbcnn-validation.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
