from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core_models import ensure_core_pretrained_models
from app.model_registry import DownloadError, ModelManifest, download_model, inspect_model
from app.paths import models_root


OPENCV_NAFNET = ModelManifest(
    key="opencv_nafnet_deblur",
    title="OpenCV Zoo NAFNet deblurring (2025may)",
    filename="deblurring_nafnet_2025may.onnx",
    destination="models/opencv_zoo/deblurring_nafnet_2025may.onnx",
    source_url="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/deblurring_nafnet/deblurring_nafnet_2025may.onnx",
    code_license="MIT",
    weights_license="MIT (OpenCV Zoo model directory)",
    conservative_default=True,
    expected_sha256="07263f416febecce10193dd648e950b22e397cf521eedab1a114ef77b2bc9587",
    max_bytes=100_000_000,
    notes=(
        "Official OpenCV Zoo NAFNet ONNX checkpoint. The Git-LFS object published by OpenCV Zoo "
        "declares size 91,736,251 bytes and SHA-256 07263f416f...9587. Runtime uses tiled OpenCV-DNN "
        "inference and the pipeline identity guardrail can roll back the result."
    ),
)


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


def ensure_production_pretrained_models(
    root: str | Path | None = None,
    *,
    face_timeout_seconds: int = 15,
    restoration_timeout_seconds: int = 75,
) -> ProductionModelBootstrap:
    """Ensure verified pretrained models used by the normal automatic path.

    Download failures are non-fatal: the caller receives exact errors and the
    conservative deterministic implementation remains available. Only models with
    pinned SHA-256 values are downloaded automatically.
    """
    target_root = Path(root).resolve() if root is not None else models_root().resolve()
    core = ensure_core_pretrained_models(target_root, timeout_seconds=face_timeout_seconds)
    paths = dict(core.paths)
    errors = dict(core.errors)

    manifest = OPENCV_NAFNET
    try:
        status = inspect_model(manifest, target_root)
        if bool(status["exists"]) and status.get("checksum_ok") is not False:
            paths[manifest.key] = Path(str(status["path"]))
        else:
            paths[manifest.key] = download_model(
                manifest,
                target_root,
                accept_license=True,
                timeout_seconds=restoration_timeout_seconds,
            )
    except (DownloadError, OSError, ValueError, RuntimeError) as exc:
        errors[manifest.key] = str(exc)

    return ProductionModelBootstrap(target_root, paths, errors)
