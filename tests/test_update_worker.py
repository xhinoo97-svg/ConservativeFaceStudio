from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import app.update_worker as worker_module
from app.model_catalog import all_models_by_key
from app.production_model_smoke import PRODUCTION_MODEL_KEYS
from app.version import APP_VERSION


def _manifest() -> dict[str, object]:
    catalog = all_models_by_key()
    return {
        "app": {
            "version": APP_VERSION,
            "url": "https://updates.example.test/setup.exe",
            "sha256": "a" * 64,
            "filename": "setup.exe",
            "max_bytes": 1000,
        },
        "models": [
            {
                "key": key,
                "version": Path(catalog[key].filename).stem,
                "url": f"https://updates.example.test/{catalog[key].filename}",
                "sha256": "b" * 64,
                "destination": catalog[key].destination,
                "max_bytes": catalog[key].max_bytes,
            }
            for key in PRODUCTION_MODEL_KEYS
        ],
    }


def test_update_check_does_not_reinstall_same_app_or_model_versions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(worker_module, "fetch_update_manifest", lambda url: _manifest())
    monkeypatch.setattr(worker_module, "user_data_root", lambda: tmp_path)
    monkeypatch.setattr(
        worker_module,
        "resolve_local_production_models",
        lambda **kwargs: SimpleNamespace(paths={key: tmp_path / key for key in PRODUCTION_MODEL_KEYS}),
    )
    reports: list[dict[str, object]] = []
    failures: list[str] = []
    worker = worker_module.UpdateWorker("https://updates.example.test/manifest.json")
    worker.completed.connect(reports.append)
    worker.failed.connect(failures.append)

    worker.run()

    assert failures == []
    assert reports[0]["production_pack_complete"] is True
    assert reports[0]["model_update_available"] is False
    assert reports[0]["app_update_available"] is False


def test_update_check_repairs_missing_model_even_when_version_is_current(monkeypatch, tmp_path: Path) -> None:
    missing = PRODUCTION_MODEL_KEYS[0]
    available = {key: tmp_path / key for key in PRODUCTION_MODEL_KEYS if key != missing}
    monkeypatch.setattr(worker_module, "fetch_update_manifest", lambda url: _manifest())
    monkeypatch.setattr(worker_module, "user_data_root", lambda: tmp_path)
    monkeypatch.setattr(
        worker_module,
        "resolve_local_production_models",
        lambda **kwargs: SimpleNamespace(paths=available),
    )
    reports: list[dict[str, object]] = []
    worker = worker_module.UpdateWorker("https://updates.example.test/manifest.json")
    worker.completed.connect(reports.append)

    worker.run()

    assert reports[0]["model_update_available"] is True
    assert reports[0]["model_update_count"] == 1
    assert reports[0]["model_repair_count"] == 1


def test_update_check_detects_newer_app(monkeypatch, tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["app"]["version"] = "99.0.0"
    monkeypatch.setattr(worker_module, "fetch_update_manifest", lambda url: manifest)
    monkeypatch.setattr(worker_module, "user_data_root", lambda: tmp_path)
    monkeypatch.setattr(
        worker_module,
        "resolve_local_production_models",
        lambda **kwargs: SimpleNamespace(paths={key: tmp_path / key for key in PRODUCTION_MODEL_KEYS}),
    )
    reports: list[dict[str, object]] = []
    worker = worker_module.UpdateWorker("https://updates.example.test/manifest.json")
    worker.completed.connect(reports.append)

    worker.run()

    assert reports[0]["app_update_available"] is True
