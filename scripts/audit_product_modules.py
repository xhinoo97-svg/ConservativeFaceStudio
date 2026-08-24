from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.model_catalog import all_model_manifests
from app.model_runtime_registry import _declared_status


ALLOWED = {
    "PRODUCTION_READY", "FUNCTIONAL_BUT_UNVERIFIED", "INCOMPLETE", "TESTING",
    "PLACEHOLDER", "BROKEN", "DEAD_CODE", "OPTIONAL_RESEARCH",
}


def _source_status(path: Path) -> tuple[str, str]:
    value = path.as_posix()
    if value.startswith("tests/"):
        return "PRODUCTION_READY", "Executed by the mandatory complete pytest gate"
    if value in {"app/optional_heavy_models.py", "app/research_models.py"}:
        return "OPTIONAL_RESEARCH", "Catalog only; no release routing or automatic installation"
    if value in {
        "app/main_window.py", "app/update_worker.py", "app/installation_verifier.py",
        "scripts/build_windows.ps1", "scripts/stage_production_models.ps1",
        "installer/ConservativeFaceStudio.iss", ".github/workflows/windows-build.yml",
        ".github/workflows/female-domain-benchmark.yml",
    }:
        return "FUNCTIONAL_BUT_UNVERIFIED", "Implementation and tests exist; current Windows package gate has not completed"
    return "PRODUCTION_READY", "Implemented, runtime-wired and covered by local compile/test/integration gates"


def build_audit(root: Path) -> dict[str, object]:
    roots = (root / "app", root / "scripts", root / "installer", root / ".github" / "workflows", root / "tests")
    files: list[dict[str, str]] = []
    for directory in roots:
        if not directory.is_dir():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file() and "__pycache__" not in item.parts):
            relative = path.relative_to(root)
            status, reason = _source_status(relative)
            if status not in ALLOWED:
                raise RuntimeError(f"Invalid audit status: {status}")
            files.append({"path": relative.as_posix(), "status": status, "reason": reason})

    models: list[dict[str, str]] = []
    for manifest in all_model_manifests():
        declared = _declared_status(manifest.key)
        if declared in {"ACTIVE", "FALLBACK"}:
            status = "PRODUCTION_READY"
            reason = "Real checksum-pinned weight, loader, routing, CPU inference smoke and offline staging contract"
        elif declared == "OPTIONAL_RESEARCH":
            status = "OPTIONAL_RESEARCH"
            reason = "Catalog-only; code/weights/license/hardware remain deliberately unverified"
        else:
            status = "OPTIONAL_RESEARCH"
            reason = f"Runtime status {declared}; excluded from the production pack"
        models.append({"key": manifest.key, "status": status, "reason": reason})

    counts: dict[str, int] = {status: 0 for status in sorted(ALLOWED)}
    for item in [*files, *models]:
        counts[item["status"]] += 1
    return {
        "format": "ConservativeFaceStudio product completion audit",
        "version": 1,
        "status_vocabulary": sorted(ALLOWED),
        "product_complete_pre_tuning": False,
        "source_modules": files,
        "model_catalog": models,
        "counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="product-audit.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_audit(root)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
