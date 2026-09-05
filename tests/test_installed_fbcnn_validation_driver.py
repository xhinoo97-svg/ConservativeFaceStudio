from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.fbcnn_upstream_backend import APPROVED_CHECKPOINT_SHA256
from scripts.run_installed_fbcnn_path_validation import (
    _jpeg_round_trip,
    _letterbox_portrait,
    validate_installed_block,
)


def _details() -> dict[str, object]:
    return {
        "engine": "installed-paper-quality-runtime-v1",
        "paper_quality_runtime_wired": True,
        "damage": {
            "dominant_damage_class": "JPEG_ARTIFACT",
            "dominant_confidence": 0.91,
            "admitted_class_evidence": [
                {
                    "damage_class": "JPEG_ARTIFACT",
                    "pixels": 100,
                    "admitted_fraction": 1.0,
                    "mean_confidence": 0.91,
                }
            ],
        },
        "damage_route": {"damage_kind": "JPEG_ARTIFACT"},
        "validation_model_route": {
            "damage_kind": "JPEG_ARTIFACT",
            "source_damage_class": "JPEG_ARTIFACT",
        },
        "models_actually_executed": [
            {
                "model_key": "fbcnn",
                "checkpoint_sha256": APPROVED_CHECKPOINT_SHA256,
                "execution_scope": "INSTALLED_PATH_VALIDATION_SHADOW",
                "fused_to_final": False,
            }
        ],
        "validation_model_candidates": [
            {
                "model_key": "fbcnn",
                "resource": {"post_unload": {"process_rss_bytes": 10}},
            }
        ],
        "validation_candidates_fused_to_final": False,
        "generated_pixels": 0,
        "wrong_person_final_pixels": 0,
        "provenance_violations": 0,
        "outside_authority_changed_pixels": 0,
        "block_input_changed_pixels_from_immutable_main": 0,
        "block_output_changed_pixels_from_input": 0,
        "model_execution_errors": [],
        "decision": "ABSTAIN",
        "paper_quality_trace": [
            {"stage": "DamageMaskRuntime", "status": "EXECUTED"},
            {"stage": "damage_router", "status": "EXECUTED"},
            {"stage": "model_execution", "status": "VALIDATION_EXECUTED_NOT_FUSED"},
            {"stage": "PaperQualityRuntime", "status": "EXECUTED"},
            {"stage": "provenance", "status": "VERIFIED"},
        ],
    }


def test_installed_validation_driver_uses_real_worker_not_research_vertical_slice() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_installed_fbcnn_path_validation.py"
    ).read_text(encoding="utf-8")
    assert "PipelineWorker(" in source
    assert "worker.run()" in source
    assert "run_fbcnn_vertical_slice" not in source
    assert "resolve_local_production_models" in source
    assert "inspect_paper_quality_validation_pack" in source


def test_installed_block_validator_requires_shadow_execution_and_preserved_block_input() -> None:
    main = np.full((24, 24, 3), 70, dtype=np.uint8)
    report = validate_installed_block(
        _details(),
        block_image=main.copy(),
        block_input=main.copy(),
        immutable_main=main,
    )
    assert report["damage_route"] == "JPEG_ARTIFACT"
    assert report["model_route"] == "JPEG_ARTIFACT"
    assert report["wrong_person_final_pixels"] == 0
    assert report["provenance_violations"] == 0
    assert report["healthy_pixels_changed"] == 0

    fused = _details()
    fused["validation_candidates_fused_to_final"] = True
    with pytest.raises(RuntimeError, match="fused"):
        validate_installed_block(
            fused,
            block_image=main.copy(),
            block_input=main.copy(),
            immutable_main=main,
        )


def test_installed_block_validator_separates_preexisting_context_from_block_delta() -> None:
    immutable = np.full((24, 24, 3), 70, dtype=np.uint8)
    block_input = immutable.copy()
    block_input[4:9, 7:13] = 71
    details = _details()
    details["block_input_changed_pixels_from_immutable_main"] = 30

    report = validate_installed_block(
        details,
        block_image=block_input.copy(),
        block_input=block_input,
        immutable_main=immutable,
    )

    assert report["block_input_changed_pixels_from_immutable_main"] == 30
    assert report["healthy_pixels_changed"] == 0


def test_installed_validation_uses_real_jpeg_bytes_and_bounded_canvas() -> None:
    source = np.full((240, 180, 3), 120, dtype=np.uint8)
    cv2.circle(source, (90, 100), 55, (80, 150, 210), -1)
    canvas = _letterbox_portrait(source)
    degraded, payload = _jpeg_round_trip(canvas, 10)

    assert canvas.shape == (384, 384, 3)
    assert degraded.shape == canvas.shape
    assert payload[:2] == b"\xff\xd8"
    assert not np.array_equal(degraded, canvas)
