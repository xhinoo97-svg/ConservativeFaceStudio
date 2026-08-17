from __future__ import annotations

"""Freeze a committed V4 candidate after 60/60 calibration admission.

The V4 final holdout must already be frozen before any candidate production-code
modification. This script never executes the holdout; it binds a same-HEAD
candidate, the verified production model pack, the calibration gate and the
immutable V4 manifests into one auditable candidate-freeze document.
"""

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

from scripts import freeze_face_smartphone_v4_final_holdout as final_freeze
from scripts.run_face_smartphone_baseline import production_model_paths, _sha256

CANDIDATE_ID = "face-domain-guard-v4"
# Pre-V4 production baseline. Later benchmark/workflow-only commits do not weaken
# this requirement because only the app/ diff is hashed below.
BASE_SHA = "3f7f461826af72870357c1822c1b81629121171e"


def _git(*args: str, binary: bool = False):
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=not binary,
    ).stdout


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha256(path)


def _verified_holdout_freeze() -> tuple[dict[str, Any], dict[str, str]]:
    required = ("sources.json", "cases.json", "freeze.json", "contract.json")
    for name in required:
        path = final_freeze.BENCHMARK_ROOT / name
        if not path.is_file():
            raise RuntimeError(f"V4 final holdout is not frozen: missing {name}")

    cases = final_freeze.build_cases()
    contract = final_freeze.build_contract()
    expected = final_freeze.build_freeze(cases, contract)
    actual = json.loads((final_freeze.BENCHMARK_ROOT / "freeze.json").read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError("V4 final-holdout freeze manifest drifted")

    files = {name: _file_sha(final_freeze.BENCHMARK_ROOT / name) for name in required}
    return actual, files


def build(calibration_gate: Path, model_root: Path) -> dict[str, Any]:
    gate = json.loads(calibration_gate.read_text(encoding="utf-8"))
    if gate.get("candidate_id") != CANDIDATE_ID or gate.get("accepted") is not True:
        raise RuntimeError("Calibration gate has not admitted face-domain-guard-v4")
    summary = gate.get("summary", {})
    if summary.get("admitted_cases") != 60:
        raise RuntimeError("Calibration gate does not admit all 60 frozen cases")
    if int(summary.get("restoration_passes", 0)) + int(summary.get("safe_predeclared_abstentions", 0)) != 60:
        raise RuntimeError("Calibration admission is not exactly restoration PASS plus safe predeclared ABSTAIN")
    if summary.get("unexpected_abstention_cases") != 0 or summary.get("error_cases") != 0:
        raise RuntimeError("Calibration contains an unexpected abstention or runtime error")
    if summary.get("wrong_person_final_pixels") != 0:
        raise RuntimeError("Calibration contains wrong-person final pixels")
    if summary.get("provenance_invalid_cases") != 0:
        raise RuntimeError("Calibration contains provenance violations")

    holdout_freeze, holdout_files = _verified_holdout_freeze()
    head = str(_git("rev-parse", "HEAD")).strip()
    tree = str(_git("rev-parse", "HEAD^{tree}")).strip()
    app_diff = bytes(_git("diff", "--binary", f"{BASE_SHA}..{head}", "--", "app", binary=True))
    if not app_diff:
        raise RuntimeError("V4 candidate has no production app diff from the pre-V4 baseline")
    changed_app_files = [
        line
        for line in str(_git("diff", "--name-only", f"{BASE_SHA}..{head}", "--", "app")).splitlines()
        if line
    ]
    models = production_model_paths(model_root)

    return {
        "schema_version": 2,
        "candidate_id": CANDIDATE_ID,
        "candidate_commit_sha": head,
        "candidate_tree_sha": tree,
        "candidate_source_diff_sha256": _sha256_bytes(app_diff),
        "candidate_changed_app_files": changed_app_files,
        "candidate_configuration": {
            "identity_firewall_threshold": 0.363,
            "identity_anchor_policy": "immutable-main-plus-main-bridged-trusted-references",
            "same_canvas_rule": "exact-pixel-verified-source-may-override-own-preflight-rejection-and-bridge-only-its-sface-component",
            "reference_only_cluster_rule": "never-identity-authority-without-main-or-same-canvas-bridge",
            "identity_rejected_rule": "never-observed-donor-unless-exact-same-canvas-verified",
            "final_identity_anchor_rule": "immutable-main-always-present; untrusted-raw-references-excluded",
            "no_identity_score_rule": "fail-closed; never max-empty-default-pass",
            "main_geometry_rule": "immutable-target-canvas",
            "wrong_person_final_pixels_max": 0,
            "target95_policy": "REPORT_ONLY",
            "abstention_policy": "frozen-predeclared-low-evidence-only; no-output; not-a-restoration-pass",
        },
        "calibration_gate_sha256": _file_sha(calibration_gate),
        "calibration_summary": {
            key: summary.get(key)
            for key in (
                "admitted_cases",
                "restoration_passes",
                "safe_predeclared_abstentions",
                "unexpected_abstention_cases",
                "wrong_person_final_pixels",
                "provenance_invalid_cases",
            )
        },
        "model_sha256": {key: _sha256(path) for key, path in models.items()},
        "final_holdout_freeze": holdout_freeze,
        "final_holdout_files": holdout_files,
        "frozen_after_calibration_before_final_holdout": True,
        "v3_consumed_and_not_used_for_candidate_tuning": True,
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
    print(
        json.dumps(
            {key: payload[key] for key in (
                "candidate_id",
                "candidate_commit_sha",
                "candidate_tree_sha",
                "candidate_source_diff_sha256",
            )},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
