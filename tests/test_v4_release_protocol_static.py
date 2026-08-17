from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_v4_protocol_python_files_compile() -> None:
    for relative in (
        "scripts/discover_face_smartphone_v4_sources.py",
        "scripts/freeze_face_smartphone_v4_final_holdout.py",
        "scripts/freeze_face_domain_guard_v4_candidate.py",
        "scripts/run_face_smartphone_v4_final_holdout.py",
        "app/identity_anchor_v4_policy.py",
    ):
        source = _text(relative)
        compile(source, str(ROOT / relative), "exec")


def test_consumed_v3_is_never_executed_by_release_quality() -> None:
    workflow = _text(".github/workflows/release-quality-v2.yml")
    assert "run_face_smartphone_v3_final_holdout.py" not in workflow
    assert "freeze_face_smartphone_v3_final_holdout.py --verify" in workflow
    assert "V3 final holdout is consumed" in workflow


def test_v4_final_workflow_has_single_explicit_request_trigger() -> None:
    workflow = _text(".github/workflows/v4-final-certification.yml")
    assert "release/v4-certification-request.json" in workflow
    assert "workflow_dispatch:" not in workflow
    assert "pull_request:" not in workflow


def test_v4_consumption_marker_is_persisted_before_runner_invocation() -> None:
    workflow = _text(".github/workflows/v4-final-certification.yml")
    marker = workflow.index("Persist V4 consumption marker before first holdout case")
    execute = workflow.index("Execute V4 final holdout exactly once")
    runner = workflow.index("python scripts/run_face_smartphone_v4_final_holdout.py")
    assert marker < execute < runner
    assert "CONSUMPTION_PATH: benchmarks/face-smartphone-v4-final-holdout/CONSUMED.json" in workflow
    assert "test ! -e \"$CONSUMPTION_PATH\"" in workflow
    assert "id: consume" in workflow[marker:execute]


def test_only_run_that_created_started_marker_may_update_final_marker() -> None:
    workflow = _text(".github/workflows/v4-final-certification.yml")
    final_marker = workflow.index("Persist final consumed disposition")
    final_disposition = workflow.index("Final certification disposition")
    section = workflow[final_marker:final_disposition]
    assert "if: always() && steps.consume.outcome == 'success'" in section
    assert "steps.authority.outcome == 'success'" not in section


def test_v4_workflow_pins_candidate_as_request_parent() -> None:
    workflow = _text(".github/workflows/v4-final-certification.yml")
    assert 'request_parent="$(git rev-parse HEAD^)"' in workflow
    assert 'test "$request_parent" = "$candidate_sha"' in workflow
    assert "git checkout --detach '${{ steps.request.outputs.candidate_sha }}'" in workflow


def test_v4_identity_and_provenance_regressions_are_targeted_before_full_pytest() -> None:
    workflow = _text(".github/workflows/v4-final-certification.yml")
    targeted = workflow.index("Targeted identity, provenance, MAIN and partial-reference regressions")
    full = workflow.index("Full pytest")
    section = workflow[targeted:full]
    for required in (
        "tests/test_identity_anchor_v4_policy.py",
        "tests/test_identity_anchor_v4_fail_closed.py",
        "tests/test_release_gate_wrong_person_provenance.py",
        "tests/test_v4_release_protocol_static.py",
        "tests/test_v4_controlface_source_parser.py",
    ):
        assert required in section


def test_v4_frozen_guardrails_remain_original_values() -> None:
    freeze_source = _text("scripts/freeze_face_smartphone_v4_final_holdout.py")
    assert '"sface_same_identity_minimum": 0.363' in freeze_source
    assert '"outside_region_mae_max": 8.0' in freeze_source
    assert '"wrong_person_final_pixels_max": 0' in freeze_source
    assert "FEMALE_IDENTITY_COUNT = 19" in freeze_source
    assert "TOTAL_IDENTITIES = FEMALE_IDENTITY_COUNT + CONTROL_IDENTITY_COUNT" in freeze_source
    assert "CASE_COUNT = 40" in freeze_source


def test_candidate_freeze_keeps_v4_identity_threshold_and_main_bridge_policy() -> None:
    candidate = _text("scripts/freeze_face_domain_guard_v4_candidate.py")
    assert '"identity_firewall_threshold": 0.363' in candidate
    assert '"identity_anchor_policy": "immutable-main-plus-main-bridged-trusted-references"' in candidate
    assert '"reference_only_cluster_rule": "never-identity-authority-without-main-or-same-canvas-bridge"' in candidate
    assert '"wrong_person_final_pixels_max": 0' in candidate
