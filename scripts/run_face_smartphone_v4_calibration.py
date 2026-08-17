from __future__ import annotations

"""Run the frozen 60-case V1 calibration for the V4 candidate.

The underlying materialization/evaluation code is unchanged. Only failures already
predeclared by the frozen manifest as LOW_EVIDENCE_ABSTAIN may be reclassified as a
safe abstention, and only for known identity-safety guardrail messages. The Windows
product is always same-HEAD qualified. The independent female-domain prerequisite is
required by the final one-shot gate, while calibration-only CI may run before that
separate workflow has completed.
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
from scripts.verify_same_head_female_domain import DEFAULT_REPORT as FEMALE_REPORT, verify as verify_female_domain
from scripts.verify_same_head_windows_product import verify as verify_windows_product


CANDIDATE_ID = "face-domain-guard-v4"


def run(
    output: Path,
    *,
    cache: Path,
    model_root: Path,
    offline_sources: bool = True,
    require_female_domain: bool = False,
) -> dict:
    windows_product = verify_windows_product()
    female_domain = None
    if FEMALE_REPORT.is_file():
        female_domain = verify_female_domain()
    elif require_female_domain:
        raise RuntimeError("Final V4 calibration requires same-HEAD female-domain evidence")

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
    report["same_head_female_domain"] = female_domain
    report["female_domain_required_for_this_calibration"] = bool(require_female_domain)
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
    parser.add_argument("--require-female-domain", action="store_true")
    args = parser.parse_args()
    report = run(
        Path(args.output),
        cache=Path(args.cache),
        model_root=Path(args.model_root),
        offline_sources=not args.online_sources,
        require_female_domain=args.require_female_domain,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 2 if report["summary"]["error_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
