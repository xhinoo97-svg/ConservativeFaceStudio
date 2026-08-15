from __future__ import annotations

from pathlib import Path

from app.model_registry import OFFICIAL_MODELS, inspect_model, registry_by_key


def test_realesrgan_x2plus_points_to_release_that_contains_x2plus() -> None:
    manifest = registry_by_key()["realesrgan_x2plus"]
    assert manifest.source_url is not None
    assert "/v0.2.1/RealESRGAN_x2plus.pth" in manifest.source_url


def test_inspect_model_reports_missing_without_side_effects(tmp_path: Path) -> None:
    manifest = registry_by_key()["3ddfa_mb1"]
    status = inspect_model(manifest, tmp_path)
    assert status["exists"] is False
    assert not (tmp_path / manifest.destination).exists()


def test_inspect_model_hashes_existing_file(tmp_path: Path) -> None:
    manifest = registry_by_key()["3ddfa_mb1"]
    target = tmp_path / manifest.destination
    target.parent.mkdir(parents=True)
    target.write_bytes(b"model-bytes")
    status = inspect_model(manifest, tmp_path)
    assert status["exists"] is True
    assert status["size_bytes"] == len(b"model-bytes")
    assert isinstance(status["sha256"], str) and len(status["sha256"]) == 64


def test_all_manifests_have_unique_keys() -> None:
    keys = [item.key for item in OFFICIAL_MODELS]
    assert len(keys) == len(set(keys))
