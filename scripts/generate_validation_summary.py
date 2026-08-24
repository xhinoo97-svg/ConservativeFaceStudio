from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.production_model_smoke import PRODUCTION_MODEL_KEYS
from scripts.run_release_failure_injection import REQUIRED_SCENARIOS


def _load(path: Path, expected: type) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, expected):
        raise RuntimeError(f"Invalid validation input: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_report(path: Path) -> dict[str, int]:
    report = _load(path, dict)
    cases = report.get("cases", [])
    if not isinstance(cases, list):
        raise RuntimeError(f"Cases missing from {path}")
    errors = sum(1 for item in cases if isinstance(item, dict) and item.get("error"))
    if errors:
        raise RuntimeError(f"Runtime errors in {path}: {errors}")
    applicable = sum(1 for item in cases if isinstance(item, dict) and item.get("target95_applicable") is True)
    target95_pass = sum(
        1 for item in cases
        if isinstance(item, dict) and item.get("target95_applicable") is True and item.get("target95_passed") is True
    )
    return {"completed": len(cases), "runtime_errors": errors, "target95_applicable": applicable, "target95_pass": target95_pass}


def generate_summary(root: Path, head: str, installer: Path, portable: Path) -> dict[str, object]:
    root = root.resolve()
    installer = installer.resolve()
    portable = portable.resolve()
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head.lower()):
        raise RuntimeError("A full tested Git SHA is required")
    if not installer.is_file() or not portable.is_file():
        raise RuntimeError("Installer or portable ZIP missing")

    validation = _load(root / "validation-report.json", dict)
    if validation.get("passed") is not True or validation.get("failed_cases"):
        raise RuntimeError("Conservative validation suite is not green")
    benchmark = _load(root / "benchmark-windows-ci.json", dict)
    if float(benchmark.get("total_ms", 0.0)) <= 0:
        raise RuntimeError("CPU benchmark is invalid")
    reference_counts = _load(root / "reference-count-smoke.json", dict)
    if reference_counts.get("status") != "PASS" or reference_counts.get("counts") != list(range(10)):
        raise RuntimeError("0..9 reference integration is not green")
    portable_report = _load(root / "portable-validation.json", dict)
    installed_report = _load(root / "installed-app-validation.json", dict)
    required_cli = {"--smoke-test", "--verify-installation", "--offline-test"}
    for name, report in (("portable", portable_report), ("installed", installed_report)):
        checks = report.get("checks", [])
        passed = {
            item.get("argument")
            for item in checks
            if isinstance(item, dict) and item.get("passed") is True and item.get("exit_code") == 0
        } if isinstance(checks, list) else set()
        if report.get("passed") is not True or passed != required_cli:
            raise RuntimeError(f"{name} validation is incomplete")

    registry = _load(root / "models/model-registry.json", dict)
    raw_models = registry.get("models", [])
    verified_models = {
        item.get("key")
        for item in raw_models
        if isinstance(item, dict)
        and item.get("status") in {"ACTIVE", "FALLBACK"}
        and item.get("installed") is True
        and item.get("checksum_ok") is True
    } if isinstance(raw_models, list) else set()
    if verified_models != set(PRODUCTION_MODEL_KEYS):
        raise RuntimeError("Production registry is not exactly the verified six-model pack")

    practical = _runtime_report(root / "practical-benchmark/practical-benchmark.json")
    matrix = _runtime_report(root / "practical-matrix/practical-matrix.json")
    failure_injection = _load(root / "failure-injection-summary.json", dict)
    scenario_records = failure_injection.get("scenarios", [])
    passed_scenarios = {
        item.get("scenario")
        for item in scenario_records
        if isinstance(item, dict) and item.get("passed") is True and item.get("exit_code") == 0
    } if isinstance(scenario_records, list) else set()
    if failure_injection.get("status") != "PASS" or passed_scenarios != set(REQUIRED_SCENARIOS):
        raise RuntimeError("Release failure-injection matrix is incomplete")
    return {
        "format": "ConservativeFaceStudio validation summary",
        "version": 1,
        "source_head": head.lower(),
        "product_complete_pre_tuning": True,
        "target95_policy": "REPORT_ONLY_UNTIL_TUNING",
        "gates": {
            "compile_import_pytest": "PASS_BY_BLOCKING_WORKFLOW_ORDER",
            "validation": "PASS",
            "cpu_benchmark": "PASS",
            "practical_runtime": practical,
            "extended_matrix_runtime": matrix,
            "reference_count_0_through_9": "PASS",
            "production_model_smoke": "6/6 PASS",
            "portable": "PASS",
            "offline": "PASS",
            "installer": "PASS",
            "installed_app": "PASS",
            "updater": "PASS_BY_PYTEST",
            "failure_injection": f"{len(passed_scenarios)}/{len(REQUIRED_SCENARIOS)} PASS",
        },
        "production_models": sorted(verified_models),
        "artifacts": {
            installer.name: {"size_bytes": installer.stat().st_size, "sha256": _sha256(installer)},
            portable.name: {"size_bytes": portable.stat().st_size, "sha256": _sha256(portable)},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--head", required=True)
    parser.add_argument("--installer", required=True)
    parser.add_argument("--portable", required=True)
    parser.add_argument("--output", default="validation-summary.json")
    args = parser.parse_args()
    payload = generate_summary(Path(args.root), args.head, Path(args.installer), Path(args.portable))
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
