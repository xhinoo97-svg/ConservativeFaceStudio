from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping, Sequence


EVIDENCE_TIERS = ("DEVELOPMENT", "VALIDATION", "PRODUCTION")

# Model-local gates. Product-wide readiness (Target95, final installer, V5, etc.)
# remains a separate higher-level decision in production_readiness.py.
REQUIRED_MODEL_PRODUCTION_GATES: tuple[str, ...] = (
    "official_repository_verified",
    "revision_pinned",
    "checkpoint_hash_verified",
    "code_license_compatible",
    "weights_license_compatible",
    "upstream_smoke_pass",
    "cfs_adapter_contract_pass",
    "identity_and_provenance_regressions_pass",
    "validation_benchmark_pass",
    "windows_installed_offline_pass",
    "target_hardware_resource_budget_pass",
)

# Typed refs stop an arbitrary prose string from becoming model authority. Release
# tooling must still verify that every referenced run/artifact/file really exists.
_REQUIRED_REF_PREFIXES: dict[str, tuple[str, ...]] = {
    "official_repository_verified": ("repo:",),
    "revision_pinned": ("commit:",),
    "checkpoint_hash_verified": ("checkpoint-sha256:",),
    "code_license_compatible": ("code-license-evidence:",),
    "weights_license_compatible": ("weights-license-evidence:",),
    "upstream_smoke_pass": ("upstream-smoke:",),
    "cfs_adapter_contract_pass": ("cfs-test:",),
    "identity_and_provenance_regressions_pass": ("cfs-test:",),
    "validation_benchmark_pass": ("benchmark-artifact-sha256:",),
    "windows_installed_offline_pass": ("github-run:", "artifact-sha256:", "candidate-sha:"),
    "target_hardware_resource_budget_pass": ("elitebook-evidence:", "candidate-sha:"),
}

_HEX_40_OR_64 = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ModelEvidenceGate:
    gate_id: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.gate_id not in REQUIRED_MODEL_PRODUCTION_GATES:
            raise ValueError(f"unknown model production gate: {self.gate_id}")
        if not self.evidence_refs:
            raise ValueError(f"model production gate lacks evidence: {self.gate_id}")
        if any(not isinstance(ref, str) or not ref.strip() for ref in self.evidence_refs):
            raise ValueError(f"invalid evidence ref for model production gate: {self.gate_id}")
        for prefix in _REQUIRED_REF_PREFIXES[self.gate_id]:
            if not any(ref.startswith(prefix) for ref in self.evidence_refs):
                raise ValueError(
                    f"model production gate {self.gate_id} lacks required evidence prefix {prefix}"
                )
        _validate_typed_hash_refs(self.evidence_refs)


@dataclass(frozen=True)
class ModelQualification:
    model_key: str
    evidence_tier: str
    production_qualified: bool
    evidence_refs: tuple[str, ...]
    gate_evidence: tuple[ModelEvidenceGate, ...] = ()
    attestation_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_key, str) or not self.model_key.strip():
            raise ValueError("model_key is required")
        if self.evidence_tier not in EVIDENCE_TIERS:
            raise ValueError("invalid model evidence_tier")
        if any(not isinstance(ref, str) or not ref.strip() for ref in self.evidence_refs):
            raise ValueError("model evidence_refs must contain non-empty strings")
        if self.production_qualified:
            if self.evidence_tier != "PRODUCTION":
                raise ValueError("production-qualified model must have PRODUCTION evidence")
            _validate_production_attestation(self)
        elif self.attestation_sha256 is not None or self.gate_evidence:
            raise ValueError("non-production qualification cannot carry production attestation")


def _validate_typed_hash_refs(refs: Sequence[str]) -> None:
    for ref in refs:
        if ref.startswith("commit:") or ref.startswith("candidate-sha:"):
            value = ref.split(":", 1)[1].lower()
            if not _HEX_40_OR_64.fullmatch(value):
                raise ValueError(f"invalid commit/candidate hash evidence: {ref}")
        elif ref.startswith(("checkpoint-sha256:", "artifact-sha256:", "benchmark-artifact-sha256:")):
            value = ref.split(":", 1)[1].lower()
            if not _HEX_64.fullmatch(value):
                raise ValueError(f"invalid SHA-256 evidence: {ref}")


def _gate_payload(gates: Sequence[ModelEvidenceGate]) -> list[dict[str, object]]:
    return [
        {"gate_id": gate.gate_id, "evidence_refs": list(gate.evidence_refs)}
        for gate in sorted(gates, key=lambda item: item.gate_id)
    ]


def _attestation_digest(
    model_key: str,
    evidence_refs: Sequence[str],
    gates: Sequence[ModelEvidenceGate],
) -> str:
    payload = {
        "schema_version": 1,
        "model_key": model_key,
        "evidence_tier": "PRODUCTION",
        "evidence_refs": sorted(set(evidence_refs)),
        "gates": _gate_payload(gates),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_candidate_binding(gates: Sequence[ModelEvidenceGate]) -> None:
    candidate_refs = {
        ref
        for gate in gates
        if gate.gate_id in {"windows_installed_offline_pass", "target_hardware_resource_budget_pass"}
        for ref in gate.evidence_refs
        if ref.startswith("candidate-sha:")
    }
    if len(candidate_refs) != 1:
        raise ValueError("Windows and target-hardware evidence must bind the same candidate SHA")


def _validate_production_attestation(qualification: ModelQualification) -> None:
    gates = qualification.gate_evidence
    ids = [gate.gate_id for gate in gates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate model production gate evidence")
    missing = sorted(set(REQUIRED_MODEL_PRODUCTION_GATES) - set(ids))
    extra = sorted(set(ids) - set(REQUIRED_MODEL_PRODUCTION_GATES))
    if missing or extra:
        raise ValueError(
            f"incomplete model production gate evidence: missing={missing}, extra={extra}"
        )
    _validate_candidate_binding(gates)
    flattened = {ref for gate in gates for ref in gate.evidence_refs}
    if not flattened.issubset(set(qualification.evidence_refs)):
        raise ValueError("model qualification evidence_refs do not cover gate evidence")
    expected = _attestation_digest(
        qualification.model_key,
        qualification.evidence_refs,
        gates,
    )
    if qualification.attestation_sha256 != expected:
        raise ValueError("model production attestation SHA-256 mismatch")


def build_production_model_qualification(
    model_key: str,
    gate_evidence: Mapping[str, Sequence[str]],
    *,
    extra_evidence_refs: Sequence[str] = (),
) -> ModelQualification:
    """Build a fail-closed model authority bound to complete production evidence."""
    unknown = sorted(set(gate_evidence) - set(REQUIRED_MODEL_PRODUCTION_GATES))
    missing = sorted(set(REQUIRED_MODEL_PRODUCTION_GATES) - set(gate_evidence))
    if unknown or missing:
        raise ValueError(
            f"incomplete model production gate evidence: missing={missing}, extra={unknown}"
        )
    gates = tuple(
        ModelEvidenceGate(gate_id, tuple(str(ref) for ref in gate_evidence[gate_id]))
        for gate_id in REQUIRED_MODEL_PRODUCTION_GATES
    )
    _validate_candidate_binding(gates)
    refs: list[str] = [str(ref) for ref in extra_evidence_refs]
    for gate in gates:
        refs.extend(gate.evidence_refs)
    if any(not ref.strip() for ref in refs):
        raise ValueError("model evidence_refs must contain non-empty strings")
    evidence_refs = tuple(dict.fromkeys(refs))
    digest = _attestation_digest(model_key, evidence_refs, gates)
    return ModelQualification(
        model_key=model_key,
        evidence_tier="PRODUCTION",
        production_qualified=True,
        evidence_refs=evidence_refs,
        gate_evidence=gates,
        attestation_sha256=digest,
    )


def nonproduction_model_qualification(
    model_key: str,
    evidence_tier: str,
    evidence_refs: Sequence[str] = (),
) -> ModelQualification:
    if evidence_tier not in {"DEVELOPMENT", "VALIDATION"}:
        raise ValueError("non-production qualification requires DEVELOPMENT or VALIDATION tier")
    return ModelQualification(
        model_key=model_key,
        evidence_tier=evidence_tier,
        production_qualified=False,
        evidence_refs=tuple(str(ref) for ref in evidence_refs),
    )
