from __future__ import annotations

"""Run face-domain-guard-v3 once on the independently frozen final holdout."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2

from scripts import freeze_face_smartphone_v3_final_holdout as final_freeze
from scripts import run_face_smartphone_baseline as core

CANDIDATE_ID = "face-domain-guard-v3"


def _acquire_sources_with_429_backoff(cache: Path) -> dict[str, Path]:
    delays = (0, 30, 90, 180)
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            time.sleep(delay)
        try:
            return core.acquire_sources(cache, offline=False)
        except HTTPError as exc:
            if exc.code != 429 or attempt == len(delays):
                raise
    raise RuntimeError("Final-holdout source acquisition exhausted retries")


def _current_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _verify_candidate_freeze(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("candidate_id") != CANDIDATE_ID:
        raise RuntimeError("Candidate freeze does not identify face-domain-guard-v3")
    if payload.get("candidate_commit_sha") != _current_head():
        raise RuntimeError("Candidate HEAD changed after calibration/candidate freeze")
    expected = final_freeze.build_freeze(final_freeze.build_cases())
    if payload.get("final_holdout_freeze") != expected:
        raise RuntimeError("Final-holdout freeze changed after candidate freeze")
    frozen_files = payload.get("final_holdout_files")
    if not isinstance(frozen_files, dict):
        raise RuntimeError("Candidate freeze is missing final-holdout file checksums")
    for name in ("sources.json", "cases.json", "freeze.json", "contract.json"):
        path = final_freeze.BENCHMARK_ROOT / name
        if frozen_files.get(name) != core._sha256(path):
            raise RuntimeError(f"Final-holdout file changed after candidate freeze: {name}")
    return payload


def _verify_source_dimensions(source_paths: dict[str, Path]) -> None:
    manifest = json.loads((final_freeze.BENCHMARK_ROOT / "sources.json").read_text(encoding="utf-8"))
    by_id = {item["source_id"]: item for item in manifest["sources"]}
    for source_id, path in source_paths.items():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Final-holdout clean source decode failed: {source_id}")
        height, width = image.shape[:2]
        if [width, height] != [int(v) for v in by_id[source_id]["original_dimensions"]]:
            raise RuntimeError(f"Final-holdout source dimensions changed: {source_id}")


def run(output: Path, *, cache: Path, model_root: Path, candidate_freeze: Path) -> dict:
    _verify_candidate_freeze(candidate_freeze)

    # Reuse the frozen v1 renderer/evaluator, but point every manifest/mask primitive
    # at the independently frozen v3 final holdout. No production pipeline code changes.
    original_freeze = core.freeze
    core.freeze = final_freeze
    try:
        # Explicit source acquisition happens only after the calibration and candidate
        # freeze checks above have passed.
        source_paths = _acquire_sources_with_429_backoff(cache)
        _verify_source_dimensions(source_paths)
        report = core.run_baseline(
            output,
            cache=cache,
            model_root=model_root,
            split="all",
            offline_sources=True,
            case_ids=None,
            candidate_id=CANDIDATE_ID,
        )
    finally:
        core.freeze = original_freeze

    report["production_sha"] = _current_head()
    report["split"] = "final_holdout"
    report["holdout_used_for_tuning"] = False
    report["final_holdout_one_shot_protocol"] = True
    report["candidate_freeze_sha256"] = core._sha256(candidate_freeze)
    (output / "baseline.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="face-smartphone-v3-final-holdout-result")
    parser.add_argument("--cache", default=".benchmark-cache/face-smartphone-v3-final-holdout")
    parser.add_argument("--model-root", default=".")
    parser.add_argument("--candidate-freeze", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if (output / "baseline.json").exists():
        raise RuntimeError("Refusing to overwrite an existing final-holdout result")
    report = run(
        output,
        cache=Path(args.cache),
        model_root=Path(args.model_root),
        candidate_freeze=Path(args.candidate_freeze),
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 2 if report["summary"]["error_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
