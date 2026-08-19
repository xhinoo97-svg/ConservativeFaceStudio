from __future__ import annotations

from pathlib import Path

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
