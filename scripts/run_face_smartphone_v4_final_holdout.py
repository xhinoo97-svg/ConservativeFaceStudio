from __future__ import annotations

"""Execute the independently frozen V4 final holdout exactly once for a frozen candidate.

This runner requires a pre-existing execution-authority record. The certification
workflow writes the corresponding persistent STARTED marker to the repository
*before* this script is invoked, so a failed or interrupted execution still
consumes V4 and cannot be retried as certification.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2

from scripts import freeze_face_smartphone_v4_final_holdout as final_freeze
from scripts import run_face_smartphone_baseline as core

CANDIDATE_ID = "face-domain-guard-v4"
BENCHMARK_ID = "cfs-face-smartphone-v4-final-holdout"


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
    raise RuntimeError("V4 final-holdout source acquisition exhausted retries")


def _current_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _expected_holdout_freeze() -> dict[str, Any]:
    cases = final_freeze.build_cases()
    contract = final_freeze.build_contract()
    return final_freeze.build_freeze(cases, contract)


def _verify_candidate_freeze(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("candidate_id") != CANDIDATE_ID:
        raise RuntimeError("Candidate freeze does not identify face-domain-guard-v4")
    if payload.get("candidate_commit_sha") != _current_head():
        raise RuntimeError("Candidate HEAD changed after V4 calibration/candidate freeze")

    expected = _expected_holdout_freeze()
    if payload.get("final_holdout_freeze") != expected:
        raise RuntimeError("V4 final-holdout freeze changed after candidate freeze")

    frozen_files = payload.get("final_holdout_files")
    if not isinstance(frozen_files, dict):
        raise RuntimeError("Candidate freeze is missing V4 final-holdout file checksums")
    for name in ("sources.json", "cases.json", "freeze.json", "contract.json"):
        manifest_path = final_freeze.BENCHMARK_ROOT / name
        if not manifest_path.is_file():
            raise RuntimeError(f"V4 final-holdout manifest missing: {name}")
        if frozen_files.get(name) != core._sha256(manifest_path):
            raise RuntimeError(f"V4 final-holdout file changed after candidate freeze: {name}")
    return payload


def _verify_execution_authority(path: Path, candidate_freeze: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    authority = json.loads(path.read_text(encoding="utf-8"))
    if authority.get("state") != "STARTED":
        raise RuntimeError("V4 execution authority is not in STARTED state")
    if authority.get("benchmark_id") != BENCHMARK_ID:
        raise RuntimeError("V4 execution authority benchmark mismatch")
    if authority.get("candidate_id") != CANDIDATE_ID:
        raise RuntimeError("V4 execution authority candidate-id mismatch")
    if authority.get("candidate_commit_sha") != _current_head():
        raise RuntimeError("V4 execution authority candidate SHA mismatch")
    if authority.get("candidate_commit_sha") != candidate.get("candidate_commit_sha"):
        raise RuntimeError("V4 execution authority does not match candidate freeze")
    if authority.get("candidate_freeze_sha256") != core._sha256(candidate_freeze):
        raise RuntimeError("V4 execution authority candidate-freeze checksum mismatch")
    nonce = str(authority.get("execution_nonce", "")).strip()
    if len(nonce) < 16:
        raise RuntimeError("V4 execution authority is missing a one-shot nonce")
    run_id = str(authority.get("workflow_run_id", "")).strip()
    if not run_id:
        raise RuntimeError("V4 execution authority is missing workflow_run_id")
    return authority


def _verify_source_dimensions(source_paths: dict[str, Path]) -> None:
    manifest = json.loads((final_freeze.BENCHMARK_ROOT / "sources.json").read_text(encoding="utf-8"))
    by_id = {str(item["source_id"]): item for item in manifest["sources"]}
    if set(source_paths) != set(by_id):
        raise RuntimeError("V4 acquired source registry does not exactly match frozen manifest")
    for source_id, path in source_paths.items():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"V4 final-holdout clean source decode failed: {source_id}")
        height, width = image.shape[:2]
        if [width, height] != [int(v) for v in by_id[source_id]["original_dimensions"]]:
            raise RuntimeError(f"V4 final-holdout source dimensions changed: {source_id}")


def run(
    output: Path,
    *,
    cache: Path,
    model_root: Path,
    candidate_freeze: Path,
    execution_authority: Path,
) -> dict[str, Any]:
    candidate = _verify_candidate_freeze(candidate_freeze)
    authority = _verify_execution_authority(execution_authority, candidate_freeze, candidate)

    original_freeze = core.freeze
    core.freeze = final_freeze
    try:
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
    report["execution_authority_sha256"] = core._sha256(execution_authority)
    report["workflow_run_id"] = str(authority["workflow_run_id"])
    report["execution_nonce_sha256"] = hashlib.sha256(
        str(authority["execution_nonce"]).encode("utf-8")
    ).hexdigest()
    (output / "baseline.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="face-smartphone-v4-final-holdout-result")
    parser.add_argument("--cache", default=".benchmark-cache/face-smartphone-v4-final-holdout")
    parser.add_argument("--model-root", default=".")
    parser.add_argument("--candidate-freeze", required=True)
    parser.add_argument("--execution-authority", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Refusing to overwrite or reuse a V4 final-holdout output directory")

    report = run(
        output,
        cache=Path(args.cache),
        model_root=Path(args.model_root),
        candidate_freeze=Path(args.candidate_freeze),
        execution_authority=Path(args.execution_authority),
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 2 if report["summary"]["error_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
