from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any

from app.fbcnn_upstream_backend import (
    APPROVED_CHECKPOINT_SHA256,
    APPROVED_CHECKPOINT_SIZE_BYTES,
    OFFICIAL_REPOSITORY,
    PINNED_REVISION,
    _load_checkout_metadata,
    _sha256,
)
from app.paths import models_root, runtime_root


MANIFEST_RELATIVE_PATH = Path("config/paper-quality-validation-pack.json")
LRASPP_ONNX_SIZE_BYTES = 12_879_910
LRASPP_ONNX_SHA256 = "708c7e9c074b2abf98dc95b8e74b3b76d687a63fb2a54a3e374db0bef37ae3a9"
LRASPP_OFFICIAL_REPOSITORY = "pytorch/vision"
LRASPP_PINNED_REVISION = "c6f39778e636ec40a69bdbc74386818c57a65af3"


@dataclass(frozen=True)
class PaperQualityValidationPack:
    root: Path | None
    paths: dict[str, Path]
    errors: dict[str, str]
    report: dict[str, object]

    @property
    def fbcnn_ready(self) -> bool:
        return "fbcnn" in self.paths and "fbcnn_upstream_root" in self.paths

    @property
    def damage_mask_ready(self) -> bool:
        return "lraspp_damage_mask" in self.paths

    @property
    def installed_jpeg_route_ready(self) -> bool:
        return self.fbcnn_ready and self.damage_mask_ready


def _inside(root: Path, relative: object) -> Path:
    value = Path(str(relative))
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("validation pack path must stay inside its root")
    candidate = (root / value).resolve()
    candidate.relative_to(root.resolve())
    return candidate


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / MANIFEST_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Paper Quality validation pack manifest must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Paper Quality validation pack schema")
    if payload.get("scope") != "VALIDATION_ONLY_NOT_FOR_DISTRIBUTION":
        raise ValueError("Paper Quality candidate pack must remain validation-only")
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("Paper Quality validation pack models must be a list")
    return payload


def _fbcnn_entry(payload: dict[str, Any]) -> dict[str, Any]:
    entries = [
        item
        for item in payload["models"]
        if isinstance(item, dict) and item.get("key") == "fbcnn"
    ]
    if len(entries) != 1:
        raise ValueError("Paper Quality validation pack requires exactly one FBCNN entry")
    item = entries[0]
    if item.get("production_qualified") is not False:
        raise ValueError("FBCNN validation pack must not claim production qualification")
    if item.get("official_repository") != OFFICIAL_REPOSITORY:
        raise ValueError("FBCNN validation pack repository mismatch")
    if item.get("pinned_revision") != PINNED_REVISION:
        raise ValueError("FBCNN validation pack revision mismatch")
    if item.get("checkpoint_sha256") != APPROVED_CHECKPOINT_SHA256:
        raise ValueError("FBCNN validation pack checkpoint SHA-256 mismatch")
    if int(item.get("checkpoint_size_bytes", -1)) != APPROVED_CHECKPOINT_SIZE_BYTES:
        raise ValueError("FBCNN validation pack checkpoint size mismatch")
    return item


def _lraspp_entry(payload: dict[str, Any]) -> dict[str, Any]:
    entries = [
        item
        for item in payload["models"]
        if isinstance(item, dict) and item.get("key") == "lraspp_damage_mask"
    ]
    if len(entries) != 1:
        raise ValueError("Paper Quality validation pack requires exactly one LR-ASPP entry")
    item = entries[0]
    if item.get("production_qualified") is not False:
        raise ValueError("LR-ASPP validation pack must not claim production qualification")
    if item.get("official_repository") != LRASPP_OFFICIAL_REPOSITORY:
        raise ValueError("LR-ASPP validation pack repository mismatch")
    if item.get("pinned_revision") != LRASPP_PINNED_REVISION:
        raise ValueError("LR-ASPP validation pack revision mismatch")
    if item.get("onnx_sha256") != LRASPP_ONNX_SHA256:
        raise ValueError("LR-ASPP validation pack ONNX SHA-256 mismatch")
    if int(item.get("onnx_size_bytes", -1)) != LRASPP_ONNX_SIZE_BYTES:
        raise ValueError("LR-ASPP validation pack ONNX size mismatch")
    return item


def inspect_paper_quality_validation_pack(root: str | Path) -> PaperQualityValidationPack:
    base = Path(root).resolve()
    errors: dict[str, str] = {}
    report: dict[str, object] = {
        "scope": "VALIDATION_ONLY_NOT_FOR_DISTRIBUTION",
        "production_qualified": False,
        "network_accessed": False,
    }
    try:
        payload = _load_manifest(base)
        fbcnn_item = _fbcnn_entry(payload)
        lraspp_item = _lraspp_entry(payload)
        checkpoint = _inside(base, fbcnn_item["checkpoint_relative_path"])
        upstream = _inside(base, fbcnn_item["upstream_relative_path"])
        damage_mask = _inside(base, lraspp_item["onnx_relative_path"])
        if not checkpoint.is_file():
            raise FileNotFoundError("FBCNN checkpoint missing from validation pack")
        if checkpoint.stat().st_size != APPROVED_CHECKPOINT_SIZE_BYTES:
            raise RuntimeError("FBCNN validation checkpoint byte size mismatch")
        checkpoint_sha = _sha256(checkpoint)
        if checkpoint_sha != APPROVED_CHECKPOINT_SHA256:
            raise RuntimeError("FBCNN validation checkpoint SHA-256 mismatch")
        _load_checkout_metadata(upstream)
        if not (upstream / "models" / "network_fbcnn.py").is_file():
            raise RuntimeError("FBCNN official network source missing from validation pack")
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("FBCNN validation PyTorch CPU runtime is not installed")
        if not damage_mask.is_file():
            raise FileNotFoundError("LR-ASPP ONNX missing from validation pack")
        if damage_mask.stat().st_size != LRASPP_ONNX_SIZE_BYTES:
            raise RuntimeError("LR-ASPP validation ONNX byte size mismatch")
        damage_mask_sha = _sha256(damage_mask)
        if damage_mask_sha != LRASPP_ONNX_SHA256:
            raise RuntimeError("LR-ASPP validation ONNX SHA-256 mismatch")
        if importlib.util.find_spec("onnxruntime") is None:
            raise RuntimeError("LR-ASPP validation ONNX Runtime is not installed")
        report.update(
            {
                "fbcnn_ready": True,
                "damage_mask_ready": True,
                "installed_jpeg_route_ready": True,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "damage_mask_onnx_sha256": damage_mask_sha,
                "damage_mask_onnx_size_bytes": damage_mask.stat().st_size,
                "official_repository": OFFICIAL_REPOSITORY,
                "pinned_revision": PINNED_REVISION,
            }
        )
        return PaperQualityValidationPack(
            root=base,
            paths={
                "fbcnn": checkpoint,
                "fbcnn_upstream_root": upstream,
                "lraspp_damage_mask": damage_mask,
            },
            errors={},
            report=report,
        )
    except Exception as exc:
        errors["validation_pack"] = f"{type(exc).__name__}:{exc}"
        report.update(
            {
                "fbcnn_ready": False,
                "damage_mask_ready": False,
                "installed_jpeg_route_ready": False,
                "error": errors["validation_pack"],
            }
        )
        return PaperQualityValidationPack(base, {}, errors, report)


def resolve_local_paper_quality_validation_models(
    package_root: str | Path | None = None,
    writable_root: str | Path | None = None,
) -> PaperQualityValidationPack:
    """Resolve a local candidate pack without downloads or production promotion."""
    roots = [
        Path(package_root).resolve() if package_root is not None else runtime_root().resolve(),
        Path(writable_root).resolve() if writable_root is not None else models_root().resolve(),
    ]
    inspected: list[PaperQualityValidationPack] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        manifest = root / MANIFEST_RELATIVE_PATH
        if not manifest.is_file():
            continue
        result = inspect_paper_quality_validation_pack(root)
        inspected.append(result)
        if result.installed_jpeg_route_ready:
            return result
    if inspected:
        return inspected[0]
    return PaperQualityValidationPack(
        root=None,
        paths={},
        errors={"validation_pack": "validation_pack_manifest_not_found"},
        report={
            "scope": "VALIDATION_ONLY_NOT_FOR_DISTRIBUTION",
            "production_qualified": False,
            "network_accessed": False,
            "fbcnn_ready": False,
            "damage_mask_ready": False,
            "installed_jpeg_route_ready": False,
            "error": "validation_pack_manifest_not_found",
        },
    )
