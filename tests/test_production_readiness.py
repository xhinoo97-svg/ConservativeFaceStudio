from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.production_readiness import (
    FROZEN_POLICY,
    REQUIRED_GATE_SCOPES,
    evaluate_production_readiness,
    load_production_readiness,
)
from scripts.evaluate_production_readiness import main


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "paper-quality-readiness.json"


def _payload() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_authoritative_snapshot_is_valid_and_fail_closed() -> None:
    report = load_production_readiness(MANIFEST)

    assert report.paper_quality_ready is False
    assert report.v5_launch_authorized is False
    assert "damage_router_qualified" in report.paper_quality_blockers
    assert "target95_overall_and_per_domain" in report.paper_quality_blockers
    assert "same_candidate_windows_installer" in report.paper_quality_blockers
    assert "exact_candidate_frozen" in report.v5_launch_blockers
    assert "independent_v5_holdout_prepared" in report.v5_launch_blockers


def test_snapshot_preserves_consumed_holdouts_and_frozen_thresholds() -> None:
    payload = _payload()
    assert payload["holdout_state"] == {
        "v3": "CONSUMED",
        "v4": "CONSUMED_FAIL",
        "v5": "NOT_CREATED_NOT_AUTHORIZED",
    }
    assert payload["frozen_policy"] == FROZEN_POLICY


def test_missing_gate_or_pass_without_evidence_is_rejected() -> None:
    payload = _payload()
    payload["gates"] = payload["gates"][:-1]
    with pytest.raises(ValueError, match="required readiness gates missing"):
        evaluate_production_readiness(payload)

    payload = _payload()
    damage = next(item for item in payload["gates"] if item["id"] == "damage_router_qualified")
    damage["status"] = "PASS"
    damage["evidence_refs"] = []
    with pytest.raises(ValueError, match="PASS gate lacks evidence"):
        evaluate_production_readiness(payload)


def test_threshold_or_scope_changes_are_rejected() -> None:
    payload = _payload()
    payload["frozen_policy"]["sface_min"] = 0.1
    with pytest.raises(ValueError, match="frozen safety policy mismatch"):
        evaluate_production_readiness(payload)

    payload = _payload()
    payload["gates"][0]["required_for"] = ["PAPER_QUALITY"]
    with pytest.raises(ValueError, match="required scopes mismatch"):
        evaluate_production_readiness(payload)


def test_only_explicit_evidence_backed_passes_can_authorize_synthetic_snapshot() -> None:
    payload = copy.deepcopy(_payload())
    for gate in payload["gates"]:
        gate["status"] = "PASS"
        gate["evidence_refs"] = [f"synthetic-dev:{gate['id']}"]

    report = evaluate_production_readiness(payload)
    assert set(REQUIRED_GATE_SCOPES) == {gate.gate_id for gate in report.gates}
    assert report.paper_quality_ready is True
    assert report.v5_launch_authorized is True


def test_cli_validates_snapshot_but_require_ready_returns_nonzero(tmp_path: Path) -> None:
    output = tmp_path / "readiness-report.json"
    assert main(["--manifest", str(MANIFEST), "--output", str(output)]) == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["paper_quality_ready"] is False
    assert saved["v5_launch_authorized"] is False
    assert main(["--manifest", str(MANIFEST), "--require-ready", "PAPER_QUALITY"]) == 2
    assert main(["--manifest", str(MANIFEST), "--require-ready", "V5_LAUNCH"]) == 2
