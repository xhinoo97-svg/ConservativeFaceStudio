from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


PAPER_QUALITY = "PAPER_QUALITY"
V5_LAUNCH = "V5_LAUNCH"
SCOPES = (PAPER_QUALITY, V5_LAUNCH)

ALLOWED_STATUSES = {
    "PASS",
    "FAIL",
    "BLOCKED",
    "NOT_VERIFIED",
    "NOT_MEASURED",
    "NOT_RUN",
}

REQUIRED_GATE_SCOPES: dict[str, tuple[str, ...]] = {
    "generic_runner_dev_e2e": (PAPER_QUALITY, V5_LAUNCH),
    "one_shot_dev_mock": (V5_LAUNCH,),
    "damage_router_qualified": (PAPER_QUALITY, V5_LAUNCH),
    "restorer_pack_qualified": (PAPER_QUALITY, V5_LAUNCH),
    "checkpoint_licenses_compatible": (PAPER_QUALITY, V5_LAUNCH),
    "offline_runtime_reproducible": (PAPER_QUALITY, V5_LAUNCH),
    "extended_identity_disjoint_benchmark": (PAPER_QUALITY, V5_LAUNCH),
    "target95_overall_and_per_domain": (PAPER_QUALITY, V5_LAUNCH),
    "wrong_person_final_pixels_zero": (PAPER_QUALITY, V5_LAUNCH),
    "provenance_violations_zero": (PAPER_QUALITY, V5_LAUNCH),
    "healthy_region_guardrail": (PAPER_QUALITY, V5_LAUNCH),
    "same_candidate_windows_installer": (PAPER_QUALITY, V5_LAUNCH),
    "clean_windows_offline_acceptance": (PAPER_QUALITY, V5_LAUNCH),
    "elitebook_acceptance": (PAPER_QUALITY, V5_LAUNCH),
    "exact_candidate_frozen": (V5_LAUNCH,),
    "independent_v5_holdout_prepared": (V5_LAUNCH,),
}

FROZEN_POLICY = {
    "sface_min": 0.363,
    "healthy_region_mae_max": 8.0,
    "wrong_person_final_pixels_max": 0,
    "provenance_violations_max": 0,
}


@dataclass(frozen=True)
class ReadinessGate:
    gate_id: str
    status: str
    required_for: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class ProductionReadinessReport:
    manifest_sha256: str
    branch: str
    paper_quality_ready: bool
    v5_launch_authorized: bool
    paper_quality_blockers: tuple[str, ...]
    v5_launch_blockers: tuple[str, ...]
    gates: tuple[ReadinessGate, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "branch": self.branch,
            "paper_quality_ready": self.paper_quality_ready,
            "v5_launch_authorized": self.v5_launch_authorized,
            "paper_quality_blockers": list(self.paper_quality_blockers),
            "v5_launch_blockers": list(self.v5_launch_blockers),
            "gates": [asdict(gate) for gate in self.gates],
        }


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_string_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{name} must contain non-empty strings")
    return result


def evaluate_production_readiness(
    payload: Mapping[str, object],
) -> ProductionReadinessReport:
    """Validate an evidence manifest and derive fail-closed release decisions.

    Absence, unknown status, missing evidence or an altered frozen policy is a schema
    failure, never an implicit PASS. Only an explicit PASS with at least one evidence
    reference satisfies a required gate.
    """
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported readiness schema_version")
    branch = payload.get("branch")
    if branch != "integration/final-paper-quality-local":
        raise ValueError("readiness manifest is not bound to the integration branch")

    policy = _require_mapping(payload.get("frozen_policy"), name="frozen_policy")
    if dict(policy) != FROZEN_POLICY:
        raise ValueError("frozen safety policy mismatch")

    holdouts = _require_mapping(payload.get("holdout_state"), name="holdout_state")
    if holdouts.get("v3") != "CONSUMED" or holdouts.get("v4") != "CONSUMED_FAIL":
        raise ValueError("consumed V3/V4 state must remain immutable")
    if holdouts.get("v5") != "NOT_CREATED_NOT_AUTHORIZED":
        raise ValueError("this manifest cannot create or authorize V5")

    raw_gates = payload.get("gates")
    if not isinstance(raw_gates, Sequence) or isinstance(raw_gates, (str, bytes)):
        raise ValueError("gates must be an array")
    parsed: list[ReadinessGate] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_gates):
        gate = _require_mapping(raw, name=f"gates[{index}]")
        gate_id = gate.get("id")
        status = gate.get("status")
        detail = gate.get("detail")
        if not isinstance(gate_id, str) or gate_id not in REQUIRED_GATE_SCOPES:
            raise ValueError(f"unknown readiness gate: {gate_id!r}")
        if gate_id in seen:
            raise ValueError(f"duplicate readiness gate: {gate_id}")
        seen.add(gate_id)
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"invalid status for {gate_id}: {status!r}")
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError(f"detail required for {gate_id}")
        required_for = _require_string_sequence(
            gate.get("required_for"),
            name=f"{gate_id}.required_for",
        )
        if required_for != REQUIRED_GATE_SCOPES[gate_id]:
            raise ValueError(f"required scopes mismatch for {gate_id}")
        evidence_refs = _require_string_sequence(
            gate.get("evidence_refs", ()),
            name=f"{gate_id}.evidence_refs",
        )
        if status == "PASS" and not evidence_refs:
            raise ValueError(f"PASS gate lacks evidence: {gate_id}")
        parsed.append(
            ReadinessGate(
                gate_id=gate_id,
                status=str(status),
                required_for=required_for,
                evidence_refs=evidence_refs,
                detail=detail,
            )
        )

    expected = set(REQUIRED_GATE_SCOPES)
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"required readiness gates missing: {missing}")

    by_id = {gate.gate_id: gate for gate in parsed}
    blockers: dict[str, tuple[str, ...]] = {}
    for scope in SCOPES:
        blockers[scope] = tuple(
            gate_id
            for gate_id, required_scopes in REQUIRED_GATE_SCOPES.items()
            if scope in required_scopes and by_id[gate_id].status != "PASS"
        )
    return ProductionReadinessReport(
        manifest_sha256=hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
        branch=branch,
        paper_quality_ready=not blockers[PAPER_QUALITY],
        v5_launch_authorized=not blockers[V5_LAUNCH],
        paper_quality_blockers=blockers[PAPER_QUALITY],
        v5_launch_blockers=blockers[V5_LAUNCH],
        gates=tuple(parsed),
    )


def load_production_readiness(path: Path | str) -> ProductionReadinessReport:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("readiness manifest root must be an object")
    return evaluate_production_readiness(payload)
