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


def test_v4_workflow_pins_candidate_as_request_parent() -> None:
    workflow = _text(".github/workflows/v4-final-certification.yml")
    assert 'request_parent="$(git rev-parse HEAD^)"' in workflow
    assert 'test "$request_parent" = "$candidate_sha"' in workflow
    assert "git checkout --detach '${{ steps.request.outputs.candidate_sha }}'" in workflow


def test_v4_frozen_guardrails_remain_original_values() -> None:
    freeze_source = _text("scripts/freeze_face_smartphone_v4_final_holdout.py")
    assert '"sface_same_identity_minimum": 0.363' in freeze_source
    assert '"outside_region_mae_max": 8.0' in freeze_source
    assert '"wrong_person_final_pixels_max": 0' in freeze_source
    assert "FEMALE_IDENTITY_COUNT = 19" in freeze_source
    assert "TOTAL_IDENTITIES = FEMALE_IDENTITY_COUNT + CONTROL_IDENTITY_COUNT" in freeze_source
    assert "CASE_COUNT = 40" in freeze_source
