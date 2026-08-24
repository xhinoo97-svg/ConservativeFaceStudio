from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_upstream_implementation_registry import (
    DEFAULT_REGISTRY,
    load_registry,
    validate_registry,
)

DEFAULT_DESTINATION = ROOT / ".research-upstreams"


def _entry(payload: dict[str, Any], key: str) -> dict[str, Any]:
    for item in payload["implementations"]:
        if item.get("key") == key:
            return item
    raise KeyError(f"Unknown upstream implementation: {key}")


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def bootstrap(
    key: str,
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    destination_root: Path = DEFAULT_DESTINATION,
    accept_research_only: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    payload = load_registry(registry_path)
    validate_registry(payload)
    item = _entry(payload, key)
    revision = item.get("pinned_revision")
    if revision is None:
        raise RuntimeError(f"{key} is NOT_VERIFIED and has no pinned revision")
    if not accept_research_only:
        raise RuntimeError(
            "Pinned upstream checkout is research-only. Pass --accept-research-only explicitly."
        )

    repository = str(item["official_repository"])
    clone_url = str(item["clone_url"])
    revision = str(revision)
    target = destination_root / key
    commands = [
        ["git", "clone", "--no-checkout", "--filter=blob:none", clone_url, str(target)],
        ["git", "-C", str(target), "fetch", "--depth", "1", "origin", revision],
        ["git", "-C", str(target), "checkout", "--detach", revision],
    ]

    if dry_run:
        return {
            "key": key,
            "official_repository": repository,
            "pinned_revision": revision,
            "target": str(target),
            "commands": commands,
            "executed": False,
        }

    destination_root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not (target / ".git").exists():
            raise RuntimeError(f"Refusing to overwrite non-Git path: {target}")
        remote = _run(["git", "-C", str(target), "remote", "get-url", "origin"])
        if remote.rstrip("/") != clone_url.rstrip("/"):
            raise RuntimeError(f"Existing checkout has wrong origin: {remote}")
        _run(commands[1])
        _run(commands[2])
    else:
        _run(commands[0])
        _run(commands[1])
        _run(commands[2])

    actual = _run(["git", "-C", str(target), "rev-parse", "HEAD"])
    if actual != revision:
        raise RuntimeError(f"Pinned upstream checkout mismatch: {actual} != {revision}")

    metadata = {
        "format": "ConservativeFaceStudio pinned upstream checkout",
        "key": key,
        "official_repository": repository,
        "clone_url": clone_url,
        "pinned_revision": revision,
        "actual_revision": actual,
        "qualification_state": item.get("qualification_state"),
        "research_only": True,
        "architecture_reimplemented_by_cfs": False,
    }
    (target / ".cfs-upstream.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Checkout an exact official research upstream without reimplementing its architecture."
    )
    parser.add_argument("key")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--accept-research-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = bootstrap(
        args.key,
        registry_path=args.registry,
        destination_root=args.destination,
        accept_research_only=args.accept_research_only,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
