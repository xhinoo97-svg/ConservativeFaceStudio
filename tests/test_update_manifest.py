from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import scripts.generate_update_manifest as generator


def test_release_update_manifest_contains_app_and_complete_model_pack(monkeypatch, tmp_path: Path) -> None:
    manifests = {}
    keys = ("a", "b")
    for key in keys:
        payload = f"model-{key}".encode()
        path = tmp_path / "models" / f"{key}.onnx"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        manifests[key] = replace(
            next(iter(generator.all_models_by_key().values())),
            key=key,
            filename=f"{key}-1.0.onnx",
            destination=f"models/{key}.onnx",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            max_bytes=1024,
        )
    installer = tmp_path / "ConservativeFaceStudio-Setup-x64.exe"
    installer.write_bytes(b"installer")
    monkeypatch.setattr(generator, "PRODUCTION_MODEL_KEYS", keys)
    monkeypatch.setattr(generator, "all_models_by_key", lambda: manifests)

    result = generator.generate_update_manifest(
        tmp_path,
        installer,
        "https://github.com/example/project/releases/download/v0.1.0",
    )

    assert result["app"]["filename"] == installer.name
    assert result["app"]["sha256"] == hashlib.sha256(b"installer").hexdigest()
    assert {item["key"] for item in result["models"]} == set(keys)
    assert all(item["url"].startswith("https://") for item in result["models"])
