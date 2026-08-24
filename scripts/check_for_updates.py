from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.activity import is_restoration_active
from app.model_runtime_registry import build_runtime_registry
from app.production_model_smoke import PRODUCTION_MODEL_KEYS, production_smoke_tests
from app.update_manager import (
    AppUpdateEntry,
    AppUpdater,
    ModelUpdateEntry,
    ModelUpdater,
    UpdateError,
    fetch_update_manifest,
    is_newer_version,
)


def _remote_entries(payload: dict[str, object]) -> tuple[ModelUpdateEntry, ...]:
    raw = payload.get("models", [])
    if not isinstance(raw, list):
        raise UpdateError("Remote models manifest must be a list")
    return tuple(ModelUpdateEntry.from_dict(item) for item in raw if isinstance(item, dict))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check or atomically install verified app/model updates.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--installed-registry", default="models/model-registry.json")
    parser.add_argument("--output", default="model-update-report.json")
    parser.add_argument("--manifest-url", help="HTTPS app/model update manifest")
    parser.add_argument("--apply-model", action="append", default=[], metavar="KEY")
    parser.add_argument("--apply-production-pack", action="store_true")
    parser.add_argument("--stage-app", action="store_true")
    parser.add_argument("--current-app-version", default="0.0.0")
    parser.add_argument("--restoration-lock", default="runtime/restoration-active.lock")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    mutating = bool(args.apply_model or args.apply_production_pack or args.stage_app)
    if mutating and not args.manifest_url:
        parser.error("--manifest-url is required for update installation")

    if args.manifest_url:
        try:
            remote = fetch_update_manifest(args.manifest_url)
            entries = _remote_entries(remote)
            selected: tuple[ModelUpdateEntry, ...] = ()
            if args.apply_production_pack:
                selected = tuple(entry for entry in entries if entry.key in PRODUCTION_MODEL_KEYS)
                missing = sorted(set(PRODUCTION_MODEL_KEYS) - {entry.key for entry in selected})
                if missing:
                    raise UpdateError(f"Remote production pack is incomplete: {', '.join(missing)}")
            elif args.apply_model:
                wanted = set(args.apply_model)
                selected = tuple(entry for entry in entries if entry.key in wanted)
                missing = sorted(wanted - {entry.key for entry in selected})
                if missing:
                    raise UpdateError(f"Requested models absent from manifest: {', '.join(missing)}")

            restoration_active = lambda: is_restoration_active(root / args.restoration_lock)
            results: list[dict[str, object]] = []
            if selected:
                updater = ModelUpdater(
                    root,
                    smoke_tests=production_smoke_tests(),
                    restoration_active=restoration_active,
                )
                results.append(updater.install_pack(selected).to_dict())
            if args.stage_app:
                raw_app = remote.get("app")
                if not isinstance(raw_app, dict):
                    raise UpdateError("Remote app manifest is missing")
                app_entry = AppUpdateEntry.from_dict(raw_app)
                if not is_newer_version(app_entry.version, args.current_app_version):
                    raise UpdateError("Remote app version is not newer than the installed version")
                app_updater = AppUpdater(root / "updates" / "app", restoration_active=restoration_active)
                results.append(app_updater.stage(app_entry).to_dict())

            if mutating:
                report = {"policy": "verified atomic update", "results": results}
                output = Path(args.output)
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
                print(f"updates={len(results)} report={output}")
                return 0
        except (OSError, ValueError, UpdateError) as exc:
            print(f"update_error={exc}")
            return 1

    current = build_runtime_registry(root)
    installed_path = (root / args.installed_registry).resolve()
    installed = {}
    if installed_path.is_file():
        try:
            installed = json.loads(installed_path.read_text(encoding="utf-8"))
        except Exception as exc:
            installed = {"read_error": str(exc)}

    old_models = {}
    if isinstance(installed, dict):
        raw = installed.get("models", [])
        if isinstance(raw, list):
            old_models = {str(item.get("key")): item for item in raw if isinstance(item, dict) and item.get("key")}
    elif isinstance(installed, list):
        old_models = {str(item.get("key")): item for item in installed if isinstance(item, dict) and item.get("key")}

    changes = []
    for item in current["models"]:
        key = item["key"]
        previous = old_models.get(key)
        if previous is None:
            changes.append({"key": key, "change": "catalog_entry_new", "action": "TEST_ONLY_DO_NOT_PROMOTE"})
            continue
        old_sha = previous.get("sha256_expected") or previous.get("expected_sha256")
        new_sha = item.get("sha256_expected")
        old_version = previous.get("version") or previous.get("filename")
        new_version = item.get("version")
        if old_sha != new_sha or old_version != new_version:
            changes.append({
                "key": key,
                "change": "catalog_version_or_hash_changed",
                "old_version": old_version,
                "new_version": new_version,
                "old_sha256": old_sha,
                "new_sha256": new_sha,
                "action": "STAGE_SMOKE_BENCHMARK_COMPARE_BEFORE_PROMOTION",
            })

    broken = [
        {"key": item["key"], "installed": item["installed"], "checksum_ok": item["checksum_ok"]}
        for item in current["models"]
        if item["status"] in {"ACTIVE", "FALLBACK"}
        and (not item["installed"] or item["checksum_ok"] is False)
    ]

    report = {
        "policy": "check only unless an explicit --apply-model/--apply-production-pack/--stage-app action is used",
        "promotion_rule": "stage -> checksum -> real smoke -> atomic activation -> rollback on failure",
        "changes": changes,
        "active_or_fallback_integrity_problems": broken,
        "current_registry": current,
    }
    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"changes={len(changes)} integrity_problems={len(broken)} report={output}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
