from __future__ import annotations

import argparse
import json
from pathlib import Path

from fbcnn_degradation_matrix import write_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = write_summary(args.report, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
