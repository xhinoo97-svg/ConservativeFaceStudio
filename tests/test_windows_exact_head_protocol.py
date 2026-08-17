from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_windows_build_is_pinned_to_exact_pr_head() -> None:
    workflow = _workflow("windows-build.yml")
    assert "CANDIDATE_SHA: ${{ github.event.pull_request.head.sha || github.sha }}" in workflow
    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in workflow
    assert "Verify exact candidate checkout" in workflow
    assert "git rev-parse HEAD" in workflow


def test_windows_validation_summary_uses_candidate_sha_not_merge_sha() -> None:
    workflow = _workflow("windows-build.yml")
    marker = "python scripts/generate_validation_summary.py"
    line = next(item for item in workflow.splitlines() if marker in item)
    assert "--head $env:CANDIDATE_SHA" in line
    assert "$env:GITHUB_SHA" not in line


def test_female_domain_is_pinned_to_same_exact_pr_head() -> None:
    workflow = _workflow("female-domain-benchmark.yml")
    assert "CANDIDATE_SHA: ${{ github.event.pull_request.head.sha || github.sha }}" in workflow
    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in workflow
    assert "Verify exact candidate checkout" in workflow
    assert "git rev-parse HEAD" in workflow
