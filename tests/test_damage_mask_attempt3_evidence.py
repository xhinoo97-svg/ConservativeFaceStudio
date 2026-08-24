from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "config" / "damage-mask-attempt3-evidence.json"
READINESS = ROOT / "config" / "paper-quality-readiness.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_attempt3_is_bound_to_the_recovered_artifact() -> None:
    payload = _read(EVIDENCE)
    assert payload["attempt"] == 3
    assert payload["workflow"] == {
        "run_id": 32087249287,
        "head_branch": "research/paper-quality-local-v2",
        "head_sha": "08df3163f317fe6a16571337178168d9845a749c",
        "conclusion": "SUCCESS",
    }
    artifact = payload["artifact"]
    assert artifact["id"] == 9307331508
    assert artifact["size_bytes"] == 2101877
    assert artifact["archive_sha256"] == (
        "e3b7aa05bcfdc9d48a803595218727f84bc255a4095caab84239c1850f7b52b8"
    )
    assert artifact["file_count"] == 5


def test_attempt3_checkpoint_export_parity_and_cpu_inference_passed() -> None:
    payload = _read(EVIDENCE)
    assert payload["model"]["checkpoint_sha256"] == (
        "e3b05272782aded20f209ddd39a3ac847cf4f3a90e5e3f02b63cae90474e2b7d"
    )
    assert payload["model"]["onnx_sha256"] == (
        "64e032d8693edc55d69a0a77d8665034d4edbeff43a93b6a622c4639a0d018c7"
    )
    runtime = payload["runtime"]
    assert runtime["onnx_argmax_segmentation_equal"] is True
    assert runtime["onnxruntime_cpu_seconds_single_face_first_call"] > 0
    assert runtime["process_ram_fraction"] < 0.8
    assert payload["infrastructure_result"] == (
        "PASS_CHECKPOINT_EXPORT_ONNX_PARITY_CPU_INFERENCE"
    )


def test_attempt3_failed_model_quality_with_six_zero_f1_classes() -> None:
    payload = _read(EVIDENCE)
    validation = payload["validation"]
    assert validation["damage_macro_f1"] == 0.17319769770764726
    assert validation["damage_macro_iou"] == 0.11302823182419285
    assert validation["zero_f1_classes"] == [
        "BLUR",
        "MOTION_BLUR",
        "PIXELATION",
        "BLOCK_MOSAIC",
        "JPEG_ARTIFACT",
        "STICKER",
    ]
    assert all(validation["per_class"][name]["f1"] == 0.0 for name in validation["zero_f1_classes"])
    assert payload["quality_result"] == "FAIL_MACRO_AND_PER_CLASS_DAMAGE_LOCALIZATION"


def test_small_unet_hypothesis_is_stopped_without_attempt4_or_refface() -> None:
    payload = _read(EVIDENCE)
    assert payload["hypothesis_status"] == "STOPPED_MODEL_DATA_QUALITY_FAIL"
    assert payload["attempt4_launched"] is False
    assert payload["production_qualified"] is False
    assert payload["refface_execution_authorized"] is False
    assert "do not launch attempt 4" in payload["exact_next_action"]

    readiness = _read(READINESS)
    gate = next(item for item in readiness["gates"] if item["id"] == "damage_router_qualified")
    assert gate["status"] == "BLOCKED"
    assert "artifact:9307331508" in gate["evidence_refs"]
