from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.core_models import CORE_MODEL_KEYS, ensure_core_pretrained_models
from app.model_registry import inspect_model, registry_by_key, sha256_path
from app.paths import models_root, runtime_root
from app.standard_pretrained import STANDARD_MODELS, ensure_standard_pretrained_models, standard_manifest_by_key


# Backward-compatible name used by existing tests/code.
OPENCV_NAFNET = standard_manifest_by_key()["opencv_nafnet_deblur"]


@dataclass(frozen=True)
class ProductionModelBootstrap:
    root: Path
    paths: dict[str, Path]
    errors: dict[str, str]

    @property
    def face_ready(self) -> bool:
        return all(key in self.paths for key in ("opencv_yunet", "opencv_sface"))

    @property
    def deblur_ready(self) -> bool:
        return "opencv_nafnet_deblur" in self.paths

    @property
    def semantic_ready(self) -> bool:
        return "face_parsing_resnet18_onnx" in self.paths

    @property
    def pose_ready(self) -> bool:
        return "head_pose_mobilenetv2_onnx" in self.paths

    @property
    def inpaint_ready(self) -> bool:
        return "opencv_lama_inpaint" in self.paths

    @property
    def standard_ready(self) -> bool:
        # "standard" means all strict/non-generative production models are present.
        # LaMa is intentionally excluded because it is a generated-pixel fallback.
        return self.deblur_ready and self.semantic_ready and self.pose_ready


def _verified_bundled_models(root: Path) -> dict[str, Path]:
    """Return only checksum-verified production models shipped beside the executable.

    The Windows installer places pretrained checkpoints under ``<app>/models``.  We
    must prefer those files before attempting a network download; otherwise an offline
    first run would silently fall back to weaker algorithms despite carrying verified
    models in the installer.
    """
    registry = registry_by_key()
    manifests = {item.key: item for item in STANDARD_MODELS}
    for key in CORE_MODEL_KEYS:
        manifests[key] = registry[key]

    paths: dict[str, Path] = {}
    for key, manifest in manifests.items():
        status = inspect_model(manifest, root)
        if bool(status.get("exists")) and status.get("checksum_ok") is not False:
            paths[key] = Path(str(status["path"]))
    return paths


def _verified_updated_models(root: Path) -> dict[str, Path]:
    """Load only updater-activated checkpoints whose file still matches its SHA-256."""
    state_path = root / "models" / "model-update-state.json"
    if not state_path.is_file():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    raw_models = payload.get("models", {}) if isinstance(payload, dict) else {}
    if not isinstance(raw_models, dict):
        return {}
    allowed = set(CORE_MODEL_KEYS) | {item.key for item in STANDARD_MODELS}
    verified: dict[str, Path] = {}
    for key, item in raw_models.items():
        if key not in allowed or not isinstance(item, dict) or item.get("status") != "ACTIVE_VERIFIED":
            continue
        destination = item.get("destination")
        expected = str(item.get("sha256") or "").lower()
        if not isinstance(destination, str) or len(expected) != 64:
            continue
        candidate = (root / destination).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        if candidate.is_file() and sha256_path(candidate).lower() == expected:
            verified[str(key)] = candidate
    return verified


def ensure_production_pretrained_models(
    root: str | Path | None = None,
    *,
    face_timeout_seconds: int = 15,
    restoration_timeout_seconds: int = 90,
) -> ProductionModelBootstrap:
    """Ensure verified pretrained production models.

    Packaged, checksum-verified models are preferred so the Windows application can
    perform its full pretrained path on first launch without internet access.  When a
    package is incomplete, missing models are downloaded sequentially into the writable
    per-user model directory and SHA-256 checked.  Strict fallbacks remain available if
    a download fails.
    """
    explicit_root = Path(root).resolve() if root is not None else None
    target_root = explicit_root if explicit_root is not None else models_root().resolve()

    bundled_paths: dict[str, Path] = {}
    updated_paths = _verified_updated_models(target_root)
    if explicit_root is None:
        bundled_paths = _verified_bundled_models(runtime_root().resolve())
        required = set(CORE_MODEL_KEYS) | {item.key for item in STANDARD_MODELS}
        selected = dict(bundled_paths)
        selected.update(updated_paths)
        if required.issubset(selected):
            selected_root = target_root if updated_paths else runtime_root().resolve()
            return ProductionModelBootstrap(selected_root, selected, {})

    core = ensure_core_pretrained_models(target_root, timeout_seconds=face_timeout_seconds)
    paths = dict(core.paths)
    errors = dict(core.errors)

    standard = ensure_standard_pretrained_models(
        target_root,
        timeout_seconds=restoration_timeout_seconds,
    )
    paths.update(standard.paths)
    errors.update(standard.errors)

    # Prefer immutable bundled files when they exist; this also avoids treating a
    # transient user-cache problem as loss of a production model.
    paths.update(bundled_paths)
    paths.update(updated_paths)
    for key in bundled_paths:
        errors.pop(key, None)
    return ProductionModelBootstrap(target_root, paths, errors)


PRODUCTION_MANIFESTS = STANDARD_MODELS


def resolve_local_production_models(
    package_root: str | Path | None = None,
    writable_root: str | Path | None = None,
) -> ProductionModelBootstrap:
    """Resolve the complete verified pack without opening the network.

    Normal restoration uses this path. Downloads are restricted to the explicit
    bootstrap/update tools, so a missing or corrupt installation cannot silently turn
    first use into an online operation.
    """
    package = Path(package_root).resolve() if package_root is not None else runtime_root().resolve()
    writable = Path(writable_root).resolve() if writable_root is not None else models_root().resolve()
    selected = _verified_bundled_models(package)
    selected.update(_verified_bundled_models(writable))
    selected.update(_verified_updated_models(writable))
    required = set(CORE_MODEL_KEYS) | {item.key for item in STANDARD_MODELS}
    errors = {key: "missing_or_checksum_invalid_offline_model" for key in sorted(required - set(selected))}
    return ProductionModelBootstrap(package, selected, errors)
