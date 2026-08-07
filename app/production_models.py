from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core_models import ensure_core_pretrained_models
from app.paths import models_root
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


def ensure_production_pretrained_models(
    root: str | Path | None = None,
    *,
    face_timeout_seconds: int = 15,
    restoration_timeout_seconds: int = 90,
) -> ProductionModelBootstrap:
    """Ensure verified pretrained production models.

    Downloads are sequential and SHA-256 checked. The normal strict path remains
    usable when a model cannot be downloaded. LaMa may be present, but is never
    treated as a conservative source of truth.
    """
    target_root = Path(root).resolve() if root is not None else models_root().resolve()
    core = ensure_core_pretrained_models(target_root, timeout_seconds=face_timeout_seconds)
    paths = dict(core.paths)
    errors = dict(core.errors)

    standard = ensure_standard_pretrained_models(
        target_root,
        timeout_seconds=restoration_timeout_seconds,
    )
    paths.update(standard.paths)
    errors.update(standard.errors)
    return ProductionModelBootstrap(target_root, paths, errors)


PRODUCTION_MANIFESTS = STANDARD_MODELS
