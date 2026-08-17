from __future__ import annotations

"""Verify that committed V4 benchmark manifests never changed after introduction."""

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "face-smartphone-v4-final-holdout"
REQUIRED = ("sources.json", "cases.json", "freeze.json", "contract.json")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name in REQUIRED:
        path = BENCHMARK / name
        if not path.is_file():
            raise RuntimeError(f"Missing frozen V4 manifest: {name}")
        relative = path.relative_to(ROOT).as_posix()
        additions = [
            line.strip()
            for line in _git("log", "--diff-filter=A", "--format=%H", "--", relative).splitlines()
            if line.strip()
        ]
        if len(additions) != 1:
            raise RuntimeError(f"Expected exactly one introduction commit for {relative}, got {len(additions)}")
        origin = additions[0]
        original_blob = _git("rev-parse", f"{origin}:{relative}")
        current_blob = _git("rev-parse", f"HEAD:{relative}")
        if original_blob != current_blob:
            raise RuntimeError(f"Frozen V4 manifest changed after introduction: {relative}")
        result[name] = {
            "introduced_commit_sha": origin,
            "introduced_blob_sha": original_blob,
            "current_blob_sha": current_blob,
        }
    return result


def main() -> int:
    print(json.dumps({"verified": True, "files": verify()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
