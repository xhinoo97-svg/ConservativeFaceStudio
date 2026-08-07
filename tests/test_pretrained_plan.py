from __future__ import annotations

from app.model_catalog import all_models_by_key
from app.pipeline import BlockKind
from app.pretrained_plan import PRETRAINED_BLOCK_PLAN, plan_by_block, validate_pretrained_plan


def test_pretrained_plan_covers_every_pipeline_block() -> None:
    validate_pretrained_plan()
    plan = plan_by_block()
    assert set(plan) == set(BlockKind)
    assert len(PRETRAINED_BLOCK_PLAN) == len(BlockKind)


def test_every_pretrained_model_key_is_registered() -> None:
    registry = all_models_by_key()
    missing = {
        model_key
        for choice in PRETRAINED_BLOCK_PLAN
        for model_key in choice.primary_models
        if model_key not in registry
    }
    assert missing == set()


def test_verified_onnx_is_primary_deblur_model() -> None:
    deblur = plan_by_block()[BlockKind.DEBLUR]
    assert deblur.primary_models[0] == "opencv_nafnet_deblur"
    assert "restormer_motion_deblur" in deblur.primary_models


def test_pretrained_semantic_and_pose_models_are_primary() -> None:
    plan = plan_by_block()
    assert plan[BlockKind.OCCLUSION_MASK].primary_models[0] == "face_parsing_resnet18_onnx"
    assert plan[BlockKind.FRONTALIZE].primary_models[0] == "head_pose_mobilenetv2_onnx"


def test_strict_reference_fusion_does_not_require_a_generative_checkpoint() -> None:
    region = plan_by_block()[BlockKind.REGION_SELECT]
    fusion = plan_by_block()[BlockKind.FUSION]
    assert "specific reference memory" in region.fallback.lower()
    assert "provenance" in fusion.fallback.lower()


def test_import_and_export_are_deterministic() -> None:
    plan = plan_by_block()
    assert plan[BlockKind.IMPORT].primary_models == ()
    assert plan[BlockKind.EXPORT].primary_models == ()
