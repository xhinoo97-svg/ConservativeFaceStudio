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
    assert settings.paper_quality_enabled is False


def test_runtime_settings_fail_closed_for_invalid_values(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"hardware_mode": "turbo", "app_update_manifest_url": "http://unsafe"}),
        encoding="utf-8",
    )

    settings = load_runtime_settings(path, tmp_path / "missing.json")

    assert settings.hardware_mode == "balanced"
    assert settings.app_update_manifest_url.startswith("https://")
    assert settings.paper_quality_enabled is False


def test_paper_quality_feature_flag_requires_a_real_boolean(tmp_path: Path) -> None:
    enabled = tmp_path / "enabled.json"
    enabled.write_text(json.dumps({"paper_quality_enabled": True}), encoding="utf-8")
    assert load_runtime_settings(enabled, tmp_path / "missing.json").paper_quality_enabled is True

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"paper_quality_enabled": "true"}), encoding="utf-8")
    assert load_runtime_settings(invalid, tmp_path / "missing.json").paper_quality_enabled is False
