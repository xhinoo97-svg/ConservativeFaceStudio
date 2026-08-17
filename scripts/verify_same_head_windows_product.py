from __future__ import annotations

"""Bind V4 admission to the exact Windows product already tested offline."""

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_ROOT = ROOT / ".same-head-release-metadata"
EXPECTED_ARTIFACTS = {
    "ConservativeFaceStudio-Setup-x64.exe",
    "ConservativeFaceStudio-Windows-x64.zip",
}
REQUIRED_PRODUCT_GATES = {
    "portable",
    "offline",
    "installer",
    "installed_app",
}


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()


def verify(metadata_root: Path = DEFAULT_METADATA_ROOT) -> dict:
    summary_path = metadata_root / "validation-summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Same-HEAD Windows validation summary missing: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if payload.get("source_head") != _head():
        raise RuntimeError("Windows product validation belongs to a different Git SHA")
    if payload.get("product_complete_pre_tuning") is not True:
        raise RuntimeError("Windows product is not marked complete before tuning")

    gates = payload.get("gates")
    if not isinstance(gates, dict):
        raise RuntimeError("Windows validation gates are missing")
    failed = sorted(name for name in REQUIRED_PRODUCT_GATES if gates.get(name) != "PASS")
    if failed:
        raise RuntimeError(f"Windows product gates not PASS: {', '.join(failed)}")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not EXPECTED_ARTIFACTS.issubset(artifacts):
        raise RuntimeError("Windows installer/portable artifacts are missing from validation summary")
    for name in sorted(EXPECTED_ARTIFACTS):
        item = artifacts.get(name)
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid Windows artifact record: {name}")
        size = int(item.get("size_bytes", 0))
        digest = str(item.get("sha256", "")).lower()
        if size < 1024 * 1024:
            raise RuntimeError(f"Windows artifact unexpectedly small: {name}")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RuntimeError(f"Windows artifact SHA-256 invalid: {name}")

    return {
        "source_head": payload["source_head"],
        "product_complete_pre_tuning": True,
        "artifacts": {name: artifacts[name] for name in sorted(EXPECTED_ARTIFACTS)},
        "gates": {name: gates[name] for name in sorted(REQUIRED_PRODUCT_GATES)},
    }


def main() -> int:
    print(json.dumps(verify(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
