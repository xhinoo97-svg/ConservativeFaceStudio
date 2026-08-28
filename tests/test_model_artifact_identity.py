from __future__ import annotations

import pytest

from app.model_artifact_identity import qualification_artifact_identity
from app.model_qualification import build_production_model_qualification


CANDIDATE_SHA = "a" * 40
CHECKPOINT_SHA = "b" * 64


def _evidence(*, repositories=("repo:fixture/official-model",)):
    return {
        "official_repository_verified": tuple(repositories),
        "revision_pinned": (f"commit:{CANDIDATE_SHA}",),
        "checkpoint_hash_verified": (f"checkpoint-sha256:{CHECKPOINT_SHA}",),
        "code_license_compatible": ("code-license-evidence:fixture",),
        "weights_license_compatible": ("weights-license-evidence:fixture",),
        "upstream_smoke_pass": ("upstream-smoke:fixture",),
        "cfs_adapter_contract_pass": ("cfs-test:adapter",),
        "identity_and_provenance_regressions_pass": ("cfs-test:identity",),
        "validation_benchmark_pass": (f"benchmark-artifact-sha256:{'c' * 64}",),
        "windows_installed_offline_pass": (
            "github-run:123",
            f"artifact-sha256:{'d' * 64}",
            f"candidate-sha:{CANDIDATE_SHA}",
        ),
        "target_hardware_resource_budget_pass": (
            "elitebook-evidence:fixture",
            f"candidate-sha:{CANDIDATE_SHA}",
        ),
    }


def test_production_qualification_resolves_one_exact_artifact_identity() -> None:
    qualification = build_production_model_qualification("fixture_model", _evidence())
    identity = qualification_artifact_identity(qualification)
    assert identity.repository == "fixture/official-model"
    assert identity.revision == CANDIDATE_SHA
    assert identity.checkpoint_sha256 == CHECKPOINT_SHA


def test_ambiguous_official_repository_evidence_fails_closed() -> None:
    qualification = build_production_model_qualification(
        "fixture_model",
        _evidence(repositories=("repo:fixture/official-model", "repo:fixture/other-model")),
    )
    with pytest.raises(ValueError, match="exactly one repo"):
        qualification_artifact_identity(qualification)
