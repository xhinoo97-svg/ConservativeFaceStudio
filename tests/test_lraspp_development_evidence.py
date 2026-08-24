from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "config" / "lraspp-development-evidence.json"
READINESS = ROOT / "config" / "paper-quality-readiness.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_lraspp_record_is_bound_to_existing_external_validation_artifact() -> None:
    payload = _read(EVIDENCE)
    assert payload["evidence_tier"] == "EXTERNAL_DEVELOPMENT_VALIDATION"
    assert payload["workflow"] == {
        "run_id": 32676602851,
        "head_branch": "research/paper-quality-local-v2",
        "head_sha": "edc5da8b55c39815cb34e10da6058ee0d2f4bc90",
        "conclusion": "SUCCESS",
    }
    artifact = payload["artifact"]
    assert artifact["id"] == 9502870418
    assert artifact["size_bytes"] == 79726
    assert artifact["archive_sha256"] == (
        "1357c983343b22f81942b130ae359a0051c0d6b79750417d17d832a89cf6b19c"
    )
    assert artifact["report_sha256"] == (
        "f20cb922dc0db0cb90b994a85228cc9ffd64fb18644fd9b5924b1f4d827ff681"
    )


def test_lraspp_data_is_development_only_identity_disjoint_and_complete() -> None:
    data = _read(EVIDENCE)["data"]
    assert data["dataset"] == "ControlFace10K"
    assert data["domain"] == "synthetic_explicit_identity"
    assert data["identity_count"] == 40
    assert data["completed_cases"] == 880
    assert data["error_cases"] == 0
    assert data["identity_disjoint_from_prior_lraspp_bank"] is True
    assert data["final_holdout_used"] is False
    assert data["training_or_tuning_authorized"] is False
    assert data["strata"]["identities_per_race_sex_stratum"] == 5


def test_lraspp_aggregate_gate_passes_but_domain_gaps_remain() -> None:
    payload = _read(EVIDENCE)
    assert payload["frozen_development_gate"]["thresholds_frozen_before_run"] is True
    assert payload["frozen_development_gate"]["aggregate_passed"] is True
    assert payload["overall"]["damage_macro_f1"] == 0.7166394224968546
    assert payload["overall"]["damage_macro_iou"] == 0.5798486772517341
    assert payload["overall"]["minimum_damage_class_f1"] == 0.3874992983834755

    asian = payload["per_domain"]["race:Asian"]
    age_50 = payload["per_domain"]["age:50"]
    assert asian["macro_f1"] < 0.7
    assert asian["minimum_class_f1"] < 0.35
    assert age_50["minimum_class_f1"] < 0.35
    assert payload["per_class"]["MOTION_BLUR"]["f1"] == 0.3874992983834755


def test_lraspp_runtime_is_cpu_offline_capable_after_acquisition() -> None:
    payload = _read(EVIDENCE)
    runtime = payload["runtime"]
    assert runtime["device"] == "onnxruntime_cpu"
    assert runtime["checkpoint_onnx_first_batch_argmax_equal"] is True
    assert runtime["network_required_after_artifact_and_dataset_acquisition"] is False
    assert runtime["max_parallel_heavy_models"] == 1
    assert runtime["process_ram_fraction"] < 0.8
    assert runtime["system_ram_fraction"] < 0.8
    assert payload["model"]["checkpoint_weights_license"] == (
        "NOT_EXPLICIT_UPSTREAM_RESEARCH_ONLY"
    )


def test_mask_only_safety_does_not_authorize_restoration_or_refface() -> None:
    payload = _read(EVIDENCE)
    safety = payload["safety_scope"]
    assert safety["model_output_type"] == "mask_logits_only"
    assert safety["can_modify_image_pixels"] is False
    assert safety["wrong_person_final_pixels"] == 0
    assert safety["provenance_violations"] == 0
    assert safety["restoration_pass_count"] == 0
    assert safety["refface_execution_authorized"] is False
    assert payload["production_qualified"] is False
    assert payload["target95"] == "NOT_MEASURED"

    readiness = _read(READINESS)
    gate = next(item for item in readiness["gates"] if item["id"] == "damage_router_qualified")
    assert gate["status"] == "BLOCKED"
    assert "artifact:9502870418" in gate["evidence_refs"]
