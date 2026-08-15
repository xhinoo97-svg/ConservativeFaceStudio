from __future__ import annotations

import json
from pathlib import Path

from app.settings import DEFAULT_UPDATE_MANIFEST_URL, load_runtime_settings


def test_runtime_settings_merge_persistent_user_override(tmp_path: Path) -> None:
    defaults = tmp_path / "defaults.json"
    user = tmp_path / "user.json"
    defaults.write_text(json.dumps({"hardware_mode": "safe"}), encoding="utf-8")
    user.write_text(json.dumps({"hardware_mode": "performance"}), encoding="utf-8")

    settings = load_runtime_settings(defaults, user)

    assert settings.hardware_mode == "performance"
    assert settings.app_update_manifest_url == DEFAULT_UPDATE_MANIFEST_URL


def test_runtime_settings_fail_closed_for_invalid_values(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"hardware_mode": "turbo", "app_update_manifest_url": "http://unsafe"}),
        encoding="utf-8",
    )

    settings = load_runtime_settings(path, tmp_path / "missing.json")

    assert settings.hardware_mode == "balanced"
    assert settings.app_update_manifest_url.startswith("https://")
