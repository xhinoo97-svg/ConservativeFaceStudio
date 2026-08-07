from __future__ import annotations

from pathlib import Path

import app.core_models as core_models
from app.core_models import CORE_MODEL_KEYS, ensure_core_pretrained_models
from app.model_registry import registry_by_key


def test_core_models_have_pinned_hashes_and_direct_sources() -> None:
    registry = registry_by_key()
    for key in CORE_MODEL_KEYS:
        manifest = registry[key]
        assert manifest.source_url is not None
        assert manifest.expected_sha256 is not None
        assert len(manifest.expected_sha256) == 64


def test_core_bootstrap_records_download_failures_without_raising(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        core_models,
        "download_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    result = ensure_core_pretrained_models(tmp_path)
    assert result.ready is False
    assert set(result.errors) == set(CORE_MODEL_KEYS)
    assert result.paths == {}


def test_core_bootstrap_reuses_verified_existing_files(tmp_path: Path, monkeypatch) -> None:
    registry = registry_by_key()
    calls: list[str] = []

    def fake_inspect(manifest, root):
        path = Path(root) / manifest.destination
        return {
            "exists": True,
            "checksum_ok": True,
            "path": str(path),
        }

    monkeypatch.setattr(core_models, "inspect_model", fake_inspect)
    monkeypatch.setattr(
        core_models,
        "download_model",
        lambda manifest, *args, **kwargs: calls.append(manifest.key),
    )
    result = ensure_core_pretrained_models(tmp_path)
    assert result.ready is True
    assert set(result.paths) == set(CORE_MODEL_KEYS)
    assert calls == []
    assert all(key in registry for key in result.paths)
