from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core_models import CORE_MODEL_KEYS, ensure_core_pretrained_models
from app.model_registry import inspect_model, registry_by_key
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
    if explicit_root is None:
        bundled_paths = _verified_bundled_models(runtime_root().resolve())
        required = set(CORE_MODEL_KEYS) | {item.key for item in STANDARD_MODELS}
        if required.issubset(bundled_paths):
            return ProductionModelBootstrap(runtime_root().resolve(), bundled_paths, {})

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
    for key in bundled_paths:
        errors.pop(key, None)
    return ProductionModelBootstrap(target_root, paths, errors)


PRODUCTION_MANIFESTS = STANDARD_MODELS
