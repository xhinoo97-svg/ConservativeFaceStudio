from __future__ import annotations

from dataclasses import dataclass
import re

from app.model_qualification import ModelQualification


_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ModelArtifactIdentity:
    repository: str
    revision: str
    checkpoint_sha256: str


def _unique_prefixed_ref(
    qualification: ModelQualification,
    *,
    gate_id: str,
    prefix: str,
) -> str:
    gate = next((item for item in qualification.gate_evidence if item.gate_id == gate_id), None)
    if gate is None:
        raise ValueError(f"production qualification lacks {gate_id}")
    values = tuple(
        ref[len(prefix) :].strip()
        for ref in gate.evidence_refs
        if ref.startswith(prefix)
    )
    unique = tuple(dict.fromkeys(value for value in values if value))
    if len(unique) != 1:
        raise ValueError(
            f"production qualification must contain exactly one {prefix} identity in {gate_id}"
        )
    return unique[0]


def qualification_artifact_identity(qualification: ModelQualification) -> ModelArtifactIdentity:
    """Return the exact upstream source/checkpoint identity authorized for production.

    A model key alone is not artifact identity. Production execution must resolve to one
    official repository identifier, one immutable source revision and one checkpoint
    SHA-256. Ambiguous or incomplete evidence fails closed.
    """
    if not qualification.production_qualified or qualification.evidence_tier != "PRODUCTION":
        raise ValueError("model is not production qualified")
    if not qualification.attestation_sha256:
        raise ValueError("production qualification lacks an attestation SHA-256")

    repository = _unique_prefixed_ref(
        qualification,
        gate_id="official_repository_verified",
        prefix="repo:",
    )
    revision = _unique_prefixed_ref(
        qualification,
        gate_id="revision_pinned",
        prefix="commit:",
    ).lower()
    checkpoint = _unique_prefixed_ref(
        qualification,
        gate_id="checkpoint_hash_verified",
        prefix="checkpoint-sha256:",
    ).lower()

    if "/" not in repository or repository.startswith("/") or repository.endswith("/"):
        raise ValueError("qualified official repository identity must be owner/name")
    if not _REVISION.fullmatch(revision):
        raise ValueError("qualified upstream revision must be a 40- or 64-hex immutable revision")
    if not _SHA256.fullmatch(checkpoint):
        raise ValueError("qualified checkpoint identity must be a full SHA-256")
    return ModelArtifactIdentity(repository, revision, checkpoint)


def candidate_matches_qualification(
    *,
    upstream_repository: str | None,
    upstream_revision: str | None,
    checkpoint_sha256: str | None,
    qualification: ModelQualification,
) -> tuple[bool, str]:
    """Fail-closed exact binding between generated output and production evidence."""
    try:
        expected = qualification_artifact_identity(qualification)
    except ValueError:
        return False, "generated_model_qualification_artifact_identity_invalid"

    repository = str(upstream_repository or "").strip()
    revision = str(upstream_revision or "").strip().lower()
    checkpoint = str(checkpoint_sha256 or "").strip().lower()
    if not repository or not revision or not checkpoint:
        return False, "generated_candidate_artifact_identity_missing"
    if repository != expected.repository:
        return False, "generated_candidate_repository_mismatch"
    if revision != expected.revision:
        return False, "generated_candidate_revision_mismatch"
    if checkpoint != expected.checkpoint_sha256:
        return False, "generated_candidate_checkpoint_mismatch"
    return True, "generated_candidate_artifact_identity_verified"
