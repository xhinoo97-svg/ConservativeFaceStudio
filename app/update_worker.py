from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from app.paths import user_data_root
from app.activity import is_restoration_active
from app.model_catalog import all_models_by_key
from app.production_models import resolve_local_production_models
from app.production_model_smoke import PRODUCTION_MODEL_KEYS, production_smoke_tests
from app.update_manager import (
    AppUpdateEntry,
    AppUpdater,
    ModelUpdateEntry,
    ModelUpdater,
    fetch_update_manifest,
    is_newer_version,
)
from app.version import APP_VERSION


class UpdateWorker(QObject):
    """Run network/update work outside the GUI thread."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, manifest_url: str, *, apply_updates: bool = False) -> None:
        super().__init__()
        self.manifest_url = str(manifest_url)
        self.apply_updates = bool(apply_updates)

    @Slot()
    def run(self) -> None:
        try:
            payload = fetch_update_manifest(self.manifest_url)
            raw_models = payload.get("models", [])
            entries = tuple(
                ModelUpdateEntry.from_dict(item)
                for item in raw_models
                if isinstance(item, dict) and item.get("key") in PRODUCTION_MODEL_KEYS
            ) if isinstance(raw_models, list) else ()
            missing = sorted(set(PRODUCTION_MODEL_KEYS) - {entry.key for entry in entries})
            root = user_data_root().resolve()
            current_versions = {
                key: Path(all_models_by_key()[key].filename).stem
                for key in PRODUCTION_MODEL_KEYS
            }
            local_pack = resolve_local_production_models(writable_root=root)
            repair_keys = set(PRODUCTION_MODEL_KEYS) - set(local_pack.paths)
            explicitly_updated: set[str] = set()
            state_path = root / "models" / "model-update-state.json"
            if state_path.is_file():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    raw_state = state.get("models", {}) if isinstance(state, dict) else {}
                    if isinstance(raw_state, dict):
                        for key, item in raw_state.items():
                            if key in current_versions and isinstance(item, dict) and isinstance(item.get("version"), str):
                                current_versions[key] = item["version"]
                                explicitly_updated.add(key)
                except (OSError, ValueError):
                    pass
            model_updates = tuple(
                entry for entry in entries
                if entry.key in repair_keys or (
                    is_newer_version(entry.version, current_versions.get(entry.key, "0"))
                    if entry.key in explicitly_updated
                    else entry.version != current_versions.get(entry.key)
                )
            )
            raw_app = payload.get("app")
            app_entry = AppUpdateEntry.from_dict(raw_app) if isinstance(raw_app, dict) else None
            app_available = app_entry is not None and is_newer_version(app_entry.version, APP_VERSION)

            report: dict[str, object] = {
                "manifest": payload,
                "production_model_count": len(entries),
                "production_pack_complete": not missing,
                "missing_production_models": missing,
                "model_update_available": bool(model_updates),
                "model_update_count": len(model_updates),
                "model_repair_count": len(repair_keys),
                "app_update_available": app_available,
                "app_version": None if app_entry is None else app_entry.version,
                "applied": False,
            }
            if self.apply_updates:
                if model_updates:
                    if missing:
                        raise RuntimeError(f"Production model update pack incomplete: {', '.join(missing)}")
                    model_result = ModelUpdater(
                        root,
                        smoke_tests=production_smoke_tests(),
                        restoration_active=is_restoration_active,
                    ).install_pack(model_updates)
                    report["model_result"] = model_result.to_dict()
                if app_available and app_entry is not None:
                    app_result = AppUpdater(
                        root / "updates" / "app", restoration_active=is_restoration_active
                    ).stage(app_entry)
                    report["app_result"] = app_result.to_dict()
                report["applied"] = True
            self.completed.emit(report)
        except Exception as exc:
            self.failed.emit(str(exc))
