from __future__ import annotations

import json
from pathlib import Path

import app.installation_verifier as verifier


def _make_directories(root: Path) -> None:
    for name in (
        "models/detection", "models/identity", "models/landmarks", "models/parsing",
        "models/pose", "models/deblur", "models/reference", "models/inpainting",
        "models/restoration", "models/optional", "config", "licenses", "runtime", "logs",
        "projects", "exports", "cache",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)


def test_installation_verifier_requires_both_model_metadata_files(monkeypatch, tmp_path: Path) -> None:
    _make_directories(tmp_path)
    monkeypatch.setattr(verifier, "_production_manifests", lambda: {})
    monkeypatch.setattr(verifier, "user_data_root", lambda: tmp_path / "user-data")

    report = verifier.verify_installation(tmp_path)

    assert report["ok"] is False
    assert report["metadata"]["models/model-registry.json"]["ok"] is False
    assert report["metadata"]["models/model-manifests.json"]["ok"] is False


def test_installation_metadata_must_contain_every_production_key(monkeypatch, tmp_path: Path) -> None:
    _make_directories(tmp_path)
    monkeypatch.setattr(verifier, "_production_manifests", lambda: {"required-model": object()})
    monkeypatch.setattr(verifier, "inspect_model", lambda manifest, root: {"exists": True, "checksum_ok": True})
    monkeypatch.setattr(verifier, "user_data_root", lambda: tmp_path / "user-data")
    (tmp_path / "models/model-registry.json").write_text(
        json.dumps({"models": [{"key": "other-model"}]}), encoding="utf-8"
    )
    (tmp_path / "models/model-manifests.json").write_text(
        json.dumps([{"key": "required-model"}]), encoding="utf-8"
    )

    report = verifier.verify_installation(tmp_path)

    assert report["ok"] is False
    assert report["metadata"]["models/model-registry.json"]["ok"] is False
    assert report["metadata"]["models/model-manifests.json"]["ok"] is True
