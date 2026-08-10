from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.model_runtime_registry import build_runtime_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Check installed model state without replacing stable models.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--installed-registry", default="models/model-registry.json")
    parser.add_argument("--output", default="model-update-report.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
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
        "policy": "check only; never overwrite stable model automatically",
        "promotion_rule": "stage -> checksum -> smoke -> benchmark -> compare -> explicit promotion",
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
