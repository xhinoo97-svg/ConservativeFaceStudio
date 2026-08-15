from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.model_registry import DownloadError, download_model, inspect_model, registry_by_key
from app.paths import models_root


CORE_MODEL_KEYS: tuple[str, ...] = ("opencv_yunet", "opencv_sface")


@dataclass(frozen=True)
class CoreModelBootstrap:
    root: Path
    paths: dict[str, Path]
    errors: dict[str, str]

    @property
    def ready(self) -> bool:
        return all(key in self.paths for key in CORE_MODEL_KEYS)


def ensure_core_pretrained_models(
    root: str | Path | None = None,
    *,
    timeout_seconds: int = 15,
) -> CoreModelBootstrap:
    """Install the tiny verified OpenCV Zoo core models when possible.

    Only manifests with pinned SHA-256 values are eligible. Network failures are
    recorded and returned instead of making the restoration pipeline fail.
    """
    target_root = Path(root).resolve() if root is not None else models_root().resolve()
    registry = registry_by_key()
    paths: dict[str, Path] = {}
    errors: dict[str, str] = {}

    for key in CORE_MODEL_KEYS:
        manifest = registry[key]
        try:
            status = inspect_model(manifest, target_root)
            if bool(status["exists"]) and status.get("checksum_ok") is not False:
                paths[key] = Path(str(status["path"]))
                continue
            path = download_model(
                manifest,
                target_root,
                accept_license=True,
                timeout_seconds=timeout_seconds,
            )
            paths[key] = path
        except (DownloadError, OSError, ValueError, RuntimeError) as exc:
            errors[key] = str(exc)

    return CoreModelBootstrap(target_root, paths, errors)
