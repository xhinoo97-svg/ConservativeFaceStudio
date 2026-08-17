from __future__ import annotations

"""Run the frozen 60-case V1 calibration for the V4 candidate.

The underlying materialization/evaluation code is unchanged. Only failures already
predeclared by the frozen manifest as LOW_EVIDENCE_ABSTAIN may be reclassified as a
safe abstention, and only for known identity-safety guardrail messages. V4 admission
is also bound to the same Git SHA whose Windows installer and portable package have
already passed the offline product validation workflow.
"""

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_face_smartphone_baseline as core
from scripts.face_smartphone_abstention import apply_predeclared_abstentions
from scripts.verify_same_head_windows_product import verify as verify_windows_product


CANDIDATE_ID = "face-domain-guard-v4"


def run(output: Path, *, cache: Path, model_root: Path, offline_sources: bool = True) -> dict:
    windows_product = verify_windows_product()
    report = core.run_baseline(
        output,
        cache=cache,
        model_root=model_root,
        split="calibration",
        offline_sources=offline_sources,
        case_ids=None,
        candidate_id=CANDIDATE_ID,
    )
    cases_payload = core.freeze.build_cases()
    apply_predeclared_abstentions(report, list(cases_payload["cases"]))
    report["same_head_windows_product"] = windows_product
    (output / "baseline.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="face-smartphone-v4-calibration")
    parser.add_argument("--cache", default=".benchmark-cache/face-smartphone-v1")
    parser.add_argument("--model-root", default=".")
    parser.add_argument("--online-sources", action="store_true")
    args = parser.parse_args()
    report = run(
        Path(args.output),
        cache=Path(args.cache),
        model_root=Path(args.model_root),
        offline_sources=not args.online_sources,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 2 if report["summary"]["error_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
