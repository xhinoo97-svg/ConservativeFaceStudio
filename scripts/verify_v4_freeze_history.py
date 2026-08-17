from __future__ import annotations

"""Verify V4 manifests against the exact Git blobs frozen before candidate tuning."""

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "face-smartphone-v4-final-holdout"
FREEZE_COMMIT = "ad564c9b1cd9514250eac08425d16c2414ead9fa"
FROZEN_BLOB_SHA = {
    "sources.json": "17ec36a8b4b6857c84e05e8ccc2e90f8007481fb",
    "cases.json": "a2c75a12400401b685a83d472ff4208f59702d18",
    "freeze.json": "43cc3eb2e35bd90bd7fdc2eb5d605c68ed52f18e",
    "contract.json": "972437be55e41e4228019c5ebebdc077678ae1ca",
}
REQUIRED = tuple(FROZEN_BLOB_SHA)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _optional_origin(relative: str) -> str:
    """Return the historical introduction commit when full history is available."""
    try:
        additions = [
            line.strip()
            for line in _git("log", "--diff-filter=A", "--format=%H", "--", relative).splitlines()
            if line.strip()
        ]
    except subprocess.CalledProcessError:
        return ""
    return additions[0] if len(additions) == 1 else ""


def verify() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name in REQUIRED:
        path = BENCHMARK / name
        if not path.is_file():
            raise RuntimeError(f"Missing frozen V4 manifest: {name}")
        relative = path.relative_to(ROOT).as_posix()
        current_blob = _git("rev-parse", f"HEAD:{relative}")
        expected_blob = FROZEN_BLOB_SHA[name]
        if current_blob != expected_blob:
            raise RuntimeError(
                f"Frozen V4 manifest blob drift: {relative}: {current_blob} != {expected_blob}"
            )
        origin = _optional_origin(relative)
        if origin and origin != FREEZE_COMMIT:
            raise RuntimeError(
                f"Unexpected V4 manifest introduction commit: {relative}: {origin} != {FREEZE_COMMIT}"
            )
        result[name] = {
            "frozen_commit_sha": FREEZE_COMMIT,
            "expected_frozen_blob_sha": expected_blob,
            "current_blob_sha": current_blob,
            "history_origin_if_available": origin,
        }
    return result


def main() -> int:
    print(json.dumps({"verified": True, "files": verify()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
