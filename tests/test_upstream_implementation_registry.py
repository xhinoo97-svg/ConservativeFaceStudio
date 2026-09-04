from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import pytest

from scripts.bootstrap_pinned_upstream import bootstrap
from scripts.verify_upstream_implementation_registry import load_registry, validate_registry


def test_registry_enforces_official_upstream_reuse_policy() -> None:
    report = validate_registry(load_registry())
    assert report["verified"] is True
    assert report["implementation_count"] == 9
    assert report["pinned_count"] == 5
    assert report["not_verified_count"] == 4
    assert report["all_official_upstreams_reused"] is True
    assert report["architecture_reimplementation_allowed"] is False

    payload = load_registry()
    fbcnn = next(item for item in payload["implementations"] if item["key"] == "fbcnn")
    assert fbcnn["code_license"] == "Apache-2.0"
    assert fbcnn["checkpoint"] == {
        "asset": "fbcnn_color.pth",
        "official_release": "v1.0",
        "download_url": "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth",
        "size_bytes": 287755111,
        "sha256": "8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8",
        "terms_state": "NOT_SEPARATELY_VERIFIED",
    }
    assert "48/48 cases PASS across 8 identities" in fbcnn["current_cfs_evidence"]
    assert fbcnn["qualification_state"] == "CANDIDATE"
    assert "multi_identity_validation" not in fbcnn["blockers"]
    assert "installed_path_validation" in fbcnn["blockers"]


def test_pinned_bootstrap_dry_run_uses_exact_official_revision(tmp_path: Path) -> None:
    result = bootstrap(
        "fbcnn",
        destination_root=tmp_path,
        accept_research_only=True,
        dry_run=True,
    )
    assert result["official_repository"] == "jiaxi-jiang/FBCNN"
    assert result["pinned_revision"] == "54d1831927506b3247e2d4d245abb4f4dab1a1cd"
    assert result["executed"] is False
    commands = result["commands"]
    assert commands[0][:4] == ["git", "clone", "--no-checkout", "--filter=blob:none"]
    assert commands[1][-1] == result["pinned_revision"]
    assert commands[2][-1] == result["pinned_revision"]


def test_pinned_bootstrap_direct_cli_uses_the_same_entrypoint(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "bootstrap_pinned_upstream.py"),
            "fbcnn",
            "--destination",
            str(tmp_path),
            "--accept-research-only",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["official_repository"] == "jiaxi-jiang/FBCNN"
    assert payload["pinned_revision"] == "54d1831927506b3247e2d4d245abb4f4dab1a1cd"
    assert payload["executed"] is False


def test_bootstrap_requires_explicit_research_only_acceptance(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="research-only"):
        bootstrap("gpen_bfr512", destination_root=tmp_path, dry_run=True)


def test_unpinned_upstream_cannot_be_bootstrapped(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="NOT_VERIFIED"):
        bootstrap(
            "refinefir",
            destination_root=tmp_path,
            accept_research_only=True,
            dry_run=True,
        )
