from __future__ import annotations

"""Verify the female-domain prerequisite report for the exact V4 candidate SHA."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / ".same-head-female-domain" / "female-domain-benchmark" / "female-domain-benchmark.json"


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(report_path: Path = DEFAULT_REPORT) -> dict:
    if not report_path.is_file():
        raise RuntimeError(f"Same-HEAD female-domain report missing: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if str(payload.get("source_head", "")).lower() != _head():
        raise RuntimeError("Female-domain report belongs to a different Git SHA")

    portraits = int(payload.get("portrait_count", 0))
    minimum_portraits = int(payload.get("minimum_required_portraits", 60))
    cases_per_portrait = int(payload.get("quick_profile_cases_per_portrait", 5))
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("Female-domain summary missing")
    completed_restorations = int(summary.get("completed_cases", 0))
    errors = int(summary.get("error_cases", 0))
    if portraits < max(60, minimum_portraits):
        raise RuntimeError("Female-domain report has fewer than 60 resolved portraits")

    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("Female-domain case list missing")
    executed_cases = len(cases)
    expected_floor = portraits * cases_per_portrait
    if executed_cases < max(300, expected_floor):
        raise RuntimeError(
            f"Female-domain report has insufficient executed cases: {executed_cases} < {max(300, expected_floor)}"
        )
    if errors != 0:
        raise RuntimeError(f"Female-domain report contains {errors} runtime error case(s)")

    scenarios = {str(item.get("scenario")) for item in cases if isinstance(item, dict)}
    required_degradations = {"gaussian_heavy_single", "mosaic_single"}
    if not required_degradations.issubset(scenarios):
        missing = sorted(required_degradations - scenarios)
        raise RuntimeError(f"Female-domain report missing severe blur/mosaic coverage: {', '.join(missing)}")

    return {
        "source_head": str(payload["source_head"]).lower(),
        "portrait_count": portraits,
        "executed_cases": executed_cases,
        "completed_restorations": completed_restorations,
        "error_cases": errors,
        "scenario_coverage": sorted(required_degradations),
        "report_sha256": _sha256(report_path),
    }


def main() -> int:
    print(json.dumps(verify(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
