from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.production_readiness import PAPER_QUALITY, SCOPES, V5_LAUNCH, load_production_readiness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the fail-closed Paper Quality/V5 readiness manifest.",
    )
    parser.add_argument(
        "--manifest",
        default=str(REPOSITORY_ROOT / "config" / "paper-quality-readiness.json"),
    )
    parser.add_argument("--output")
    parser.add_argument("--require-ready", choices=SCOPES)
    args = parser.parse_args(argv)

    report = load_production_readiness(args.manifest)
    payload = report.to_dict()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.require_ready == PAPER_QUALITY and not report.paper_quality_ready:
        return 2
    if args.require_ready == V5_LAUNCH and not report.v5_launch_authorized:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
