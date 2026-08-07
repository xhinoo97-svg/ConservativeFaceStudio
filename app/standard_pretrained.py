from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.model_registry import DownloadError, ModelManifest, download_model, inspect_model
from app.paths import models_root


STANDARD_MODELS: tuple[ModelManifest, ...] = (
    ModelManifest(
        key="opencv_nafnet_deblur",
        title="OpenCV Zoo NAFNet deblurring 2025may",
        filename="deblurring_nafnet_2025may.onnx",
        destination="models/nafnet/deblurring_nafnet_2025may.onnx",
        source_url=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/deblurring_nafnet/deblurring_nafnet_2025may.onnx"
        ),
        code_license="MIT",
        weights_license="MIT (OpenCV Zoo model directory)",
        conservative_default=True,
        expected_sha256="07263f416febecce10193dd648e950b22e397cf521eedab1a114ef77b2bc9587",
        max_bytes=110_000_000,
        notes=(
            "Official OpenCV Zoo ONNX checkpoint. The strict path blends the learned output with "
            "observed pixels and still applies the per-block identity rollback."
        ),
    ),
    ModelManifest(
        key="face_parsing_resnet18_onnx",
        title="Face Parsing ResNet18 ONNX",
        filename="resnet18.onnx",
        destination="models/face_parsing/resnet18.onnx",
        source_url="https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx",
        code_license="MIT",
        weights_license="MIT upstream repository",
        conservative_default=True,
        expected_sha256="0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f",
        max_bytes=65_000_000,
        notes=(
            "Pretrained on CelebAMask-HQ. Used only as semantic facial support; an occlusion is "
            "never repaired in strict mode unless real aligned references also support it."
        ),
    ),
    ModelManifest(
        key="head_pose_mobilenetv2_onnx",
        title="Head Pose MobileNetV2 ONNX",
        filename="mobilenetv2.onnx",
        destination="models/head_pose/mobilenetv2.onnx",
        source_url="https://github.com/yakhyo/head-pose-estimation/releases/download/weights/mobilenetv2.onnx",
        code_license="MIT",
        weights_license="MIT upstream repository",
        conservative_default=True,
        expected_sha256="1e902872868e483bd0e4f8f4a8ff2a4d61c2ccbca9dadf748e5479b5cc86a9e9",
        max_bytes=15_000_000,
        notes=(
            "Lightweight pretrained pose model. Strict mode uses yaw/pitch as a safety gate and "
            "only normalizes supported 2D roll; it never synthesizes an unseen side of the face."
        ),
    ),
)

STANDARD_MODEL_KEYS = tuple(item.key for item in STANDARD_MODELS)


@dataclass(frozen=True)
class StandardModelBootstrap:
    root: Path
    paths: dict[str, Path]
    errors: dict[str, str]

    @property
    def ready(self) -> bool:
        return all(key in self.paths for key in STANDARD_MODEL_KEYS)


def standard_manifest_by_key() -> dict[str, ModelManifest]:
    return {item.key: item for item in STANDARD_MODELS}


def ensure_standard_pretrained_models(
    root: str | Path | None = None,
    *,
    timeout_seconds: int = 90,
) -> StandardModelBootstrap:
    """Install the verified non-generative standard model pack.

    Downloads happen sequentially: only one model is transferred/loaded at a time.
    Missing network access is recorded rather than turning the pipeline into an error.
    """
    target_root = Path(root).resolve() if root is not None else models_root().resolve()
    paths: dict[str, Path] = {}
    errors: dict[str, str] = {}

    for manifest in STANDARD_MODELS:
        try:
            status = inspect_model(manifest, target_root)
            if bool(status["exists"]) and status.get("checksum_ok") is not False:
                paths[manifest.key] = Path(str(status["path"]))
                continue
            path = download_model(
                manifest,
                target_root,
                accept_license=True,
                timeout_seconds=timeout_seconds,
            )
            paths[manifest.key] = path
        except (DownloadError, OSError, ValueError, RuntimeError) as exc:
            errors[manifest.key] = str(exc)

    return StandardModelBootstrap(target_root, paths, errors)
