from __future__ import annotations

import json
from dataclasses import replace

import pytest

from app.model_registry import OFFICIAL_MODELS, DownloadError, download_model, export_registry, registry_by_key, validate_manifest


def test_registry_keys_are_unique() -> None:
    keys = [item.key for item in OFFICIAL_MODELS]
    assert len(keys) == len(set(keys))


def test_all_manifests_are_valid() -> None:
    for manifest in OFFICIAL_MODELS:
        validate_manifest(manifest)


def test_non_https_source_is_rejected() -> None:
    manifest = replace(OFFICIAL_MODELS[0], source_url="http://example.invalid/model.bin")
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_unsafe_destination_is_rejected() -> None:
    manifest = replace(OFFICIAL_MODELS[0], destination="../model.bin")
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_license_acceptance_is_required(tmp_path) -> None:
    with pytest.raises(PermissionError):
        download_model(OFFICIAL_MODELS[0], tmp_path, accept_license=False)


def test_download_without_pinned_checksum_is_blocked(tmp_path) -> None:
    manifest = registry_by_key()["realesrgan_x2plus"]
    assert manifest.source_url is not None
    assert manifest.expected_sha256 is None
    with pytest.raises(DownloadError, match="checksum"):
        download_model(manifest, tmp_path, accept_license=True)


def test_manual_model_has_no_automatic_download(tmp_path) -> None:
    manifest = registry_by_key()["insightface_identity"]
    with pytest.raises(DownloadError):
        download_model(manifest, tmp_path, accept_license=True)


def test_registry_export(tmp_path) -> None:
    output = tmp_path / "registry.json"
    export_registry(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload) == len(OFFICIAL_MODELS)
    assert all("weights_license" in item for item in payload)
