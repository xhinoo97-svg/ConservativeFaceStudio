from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "config" / "upstream-implementations.json"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Upstream registry must be a JSON object")
    return payload


def validate_registry(payload: dict[str, Any]) -> dict[str, Any]:
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("Registry policy is missing")
    if policy.get("architecture_reimplementation") != "forbidden_when_official_executable_upstream_exists":
        raise ValueError("Official-upstream reuse policy is not enforced")
    if policy.get("integration_mode") != "pinned_official_upstream_plus_thin_cfs_adapter":
        raise ValueError("Unexpected upstream integration mode")
    if policy.get("upstream_code_is_assumed_bug_free") is not False:
        raise ValueError("Upstream code must never be assumed bug-free")

    implementations = payload.get("implementations")
    if not isinstance(implementations, list) or not implementations:
        raise ValueError("Registry has no upstream implementations")

    keys: set[str] = set()
    repositories: set[str] = set()
    pinned = 0
    not_verified = 0
    for raw in implementations:
        if not isinstance(raw, dict):
            raise ValueError("Implementation entries must be JSON objects")
        key = str(raw.get("key", "")).strip()
        repository = str(raw.get("official_repository", "")).strip()
        clone_url = str(raw.get("clone_url", "")).strip()
        state = str(raw.get("qualification_state", "")).strip().upper()
        revision = raw.get("pinned_revision")

        if not key or key in keys:
            raise ValueError(f"Missing or duplicate implementation key: {key!r}")
        if not _REPOSITORY.fullmatch(repository) or repository in repositories:
            raise ValueError(f"Missing, malformed or duplicate repository: {repository!r}")
        if clone_url != f"https://github.com/{repository}.git":
            raise ValueError(f"Clone URL does not match official repository for {key}")
        if raw.get("repository_verified") is not True:
            raise ValueError(f"Repository has not been verified for {key}")
        if raw.get("reuse_upstream_code") is not True:
            raise ValueError(f"Official upstream reuse is disabled for {key}")
        if state not in {"CANDIDATE", "BLOCKED", "NOT_VERIFIED"}:
            raise ValueError(f"Unexpected qualification state for {key}: {state}")

        if revision is None:
            if state != "NOT_VERIFIED":
                raise ValueError(f"Unpinned implementation must remain NOT_VERIFIED: {key}")
            not_verified += 1
        else:
            revision_text = str(revision).strip().lower()
            if not _SHA40.fullmatch(revision_text):
                raise ValueError(f"Pinned revision is not a full Git SHA for {key}")
            pinned += 1

        blockers = raw.get("blockers")
        if not isinstance(blockers, list):
            raise ValueError(f"Blocker list is missing for {key}")
        keys.add(key)
        repositories.add(repository)

    return {
        "verified": True,
        "implementation_count": len(implementations),
        "pinned_count": pinned,
        "not_verified_count": not_verified,
        "all_official_upstreams_reused": True,
        "architecture_reimplementation_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    report = validate_registry(load_registry(args.registry))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
