from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.paths import runtime_root, user_data_root


DEFAULT_UPDATE_MANIFEST_URL = (
    "https://github.com/xhinoo97-svg/ConservativeFaceStudio/releases/latest/download/update-manifest.json"
)


@dataclass(frozen=True)
class RuntimeSettings:
    hardware_mode: str = "balanced"
    app_update_manifest_url: str = DEFAULT_UPDATE_MANIFEST_URL


def _read(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_runtime_settings(
    default_path: str | Path | None = None,
    user_path: str | Path | None = None,
) -> RuntimeSettings:
    """Load bounded settings; installed user overrides survive app updates."""
    defaults = Path(default_path) if default_path is not None else runtime_root() / "config/default-settings.json"
    override = Path(user_path) if user_path is not None else user_data_root() / "config/settings.json"
    payload = {**_read(defaults), **_read(override)}
    mode = str(payload.get("hardware_mode", "balanced")).strip().lower()
    if mode not in {"safe", "balanced", "performance"}:
        mode = "balanced"
    url = str(payload.get("app_update_manifest_url", DEFAULT_UPDATE_MANIFEST_URL)).strip()
    if not url.startswith("https://"):
        url = DEFAULT_UPDATE_MANIFEST_URL
    return RuntimeSettings(mode, url)
