from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.run_face_smartphone_v4_final_holdout as runner


HEAD = "a" * 40


def _candidate_freeze(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.json"
    path.write_text("{}", encoding="utf-8")
    return path


def _authority(tmp_path: Path, candidate_freeze: Path, **overrides) -> Path:
    payload = {
        "state": "STARTED",
        "benchmark_id": runner.BENCHMARK_ID,
        "candidate_id": runner.CANDIDATE_ID,
        "candidate_commit_sha": HEAD,
        "candidate_freeze_sha256": hashlib.sha256(candidate_freeze.read_bytes()).hexdigest(),
        "windows_product_run_id": "12345",
        "female_domain_run_id": "67890",
        "workflow_run_id": "11111",
        "execution_nonce": "0123456789abcdef0123456789abcdef",
    }
    payload.update(overrides)
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _patch_environment(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_current_head", lambda: HEAD)
    monkeypatch.setattr(
        runner.core,
        "_sha256",
        lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )


def test_execution_authority_requires_both_exact_sha_prerequisite_run_ids(tmp_path: Path, monkeypatch) -> None:
    _patch_environment(monkeypatch)
    candidate_freeze = _candidate_freeze(tmp_path)
    candidate = {"candidate_commit_sha": HEAD}
    authority = _authority(tmp_path, candidate_freeze)

    payload = runner._verify_execution_authority(authority, candidate_freeze, candidate)
    assert payload["windows_product_run_id"] == "12345"
    assert payload["female_domain_run_id"] == "67890"


def test_execution_authority_rejects_missing_windows_product_run(tmp_path: Path, monkeypatch) -> None:
    _patch_environment(monkeypatch)
    candidate_freeze = _candidate_freeze(tmp_path)
    authority = _authority(tmp_path, candidate_freeze, windows_product_run_id="")
    with pytest.raises(RuntimeError, match="Windows product run id"):
        runner._verify_execution_authority(authority, candidate_freeze, {"candidate_commit_sha": HEAD})


def test_execution_authority_rejects_missing_female_domain_run(tmp_path: Path, monkeypatch) -> None:
    _patch_environment(monkeypatch)
    candidate_freeze = _candidate_freeze(tmp_path)
    authority = _authority(tmp_path, candidate_freeze, female_domain_run_id="")
    with pytest.raises(RuntimeError, match="female-domain run id"):
        runner._verify_execution_authority(authority, candidate_freeze, {"candidate_commit_sha": HEAD})


def test_execution_authority_rejects_nonnumeric_prerequisite_run_ids(tmp_path: Path, monkeypatch) -> None:
    _patch_environment(monkeypatch)
    candidate_freeze = _candidate_freeze(tmp_path)
    authority = _authority(tmp_path, candidate_freeze, female_domain_run_id="not-a-run")
    with pytest.raises(RuntimeError, match="must be numeric"):
        runner._verify_execution_authority(authority, candidate_freeze, {"candidate_commit_sha": HEAD})
