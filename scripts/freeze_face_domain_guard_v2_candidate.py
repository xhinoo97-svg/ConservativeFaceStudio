from __future__ import annotations

"""Freeze the committed face-domain-guard-v2 candidate after calibration admission."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import freeze_face_smartphone_v2_final_holdout as final_freeze
from scripts.run_face_smartphone_baseline import production_model_paths, _sha256

CANDIDATE_ID = "face-domain-guard-v2"
BASE_SHA = "aebab7bbd28fd2a1fc1adda0ab0cb126109f1122"


def _git(*args: str, binary: bool = False):
    return subprocess.run(
        ["git", *args], cwd=REPOSITORY_ROOT, check=True, capture_output=True,
        text=not binary,
    ).stdout


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha256(path)


def build(calibration_gate: Path, model_root: Path) -> dict[str, Any]:
    gate = json.loads(calibration_gate.read_text(encoding="utf-8"))
    if gate.get("candidate_id") != CANDIDATE_ID or gate.get("accepted") is not True:
        raise RuntimeError("Calibration gate has not admitted face-domain-guard-v2")
    if gate.get("summary", {}).get("hard_guardrail_passes") != 60:
        raise RuntimeError("Calibration gate is not 60/60")
    if gate.get("summary", {}).get("wrong_person_final_pixels") != 0:
        raise RuntimeError("Calibration contains wrong-person final pixels")
    if gate.get("summary", {}).get("provenance_invalid_cases") != 0:
        raise RuntimeError("Calibration contains provenance violations")

    head = str(_git("rev-parse", "HEAD")).strip()
    tree = str(_git("rev-parse", "HEAD^{tree}")).strip()
    app_diff = bytes(_git("diff", "--binary", f"{BASE_SHA}..{head}", "--", "app", binary=True))
    if not app_diff:
        raise RuntimeError("Candidate has no production app diff from verified pre-v2 HEAD")
    changed_app_files = [line for line in str(_git("diff", "--name-only", f"{BASE_SHA}..{head}", "--", "app")).splitlines() if line]
    models = production_model_paths(model_root)
    holdout_cases = final_freeze.build_cases()
    holdout_freeze = final_freeze.build_freeze(holdout_cases)

    return {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "candidate_commit_sha": head,
        "candidate_tree_sha": tree,
        "candidate_source_diff_sha256": _sha256_bytes(app_diff),
        "candidate_changed_app_files": changed_app_files,
        "candidate_configuration": {
            "identity_firewall": "preflight-per-source-v2",
            "identity_rejected_rule": "never_observed_donor",
            "partial_identity_unknown_rule": "existing_strict_partial_component_path_only",
            "preservation_seed": "precise_same_canvas_damage_seed",
            "target95_policy": "REPORT_ONLY",
        },
        "calibration_gate_sha256": _file_sha(calibration_gate),
        "model_sha256": {key: _sha256(path) for key, path in models.items()},
        "final_holdout_freeze": holdout_freeze,
        "final_holdout_files": {
            name: _file_sha(final_freeze.BENCHMARK_ROOT / name)
            for name in ("sources.json", "cases.json", "freeze.json", "contract.json")
        },
        "frozen_after_calibration_before_final_holdout": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-gate", required=True)
    parser.add_argument("--model-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build(Path(args.calibration_gate), Path(args.model_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("candidate_id", "candidate_commit_sha", "candidate_tree_sha", "candidate_source_diff_sha256")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
