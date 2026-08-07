from __future__ import annotations

from app.model_registry import registry_by_key
from app.pipeline import BlockKind
from app.pretrained_plan import PRETRAINED_BLOCK_PLAN, plan_by_block, validate_pretrained_plan


def test_pretrained_plan_covers_every_pipeline_block() -> None:
    validate_pretrained_plan()
    plan = plan_by_block()
    assert set(plan) == set(BlockKind)
    assert len(PRETRAINED_BLOCK_PLAN) == len(BlockKind)


def test_every_pretrained_model_key_is_registered() -> None:
    registry = registry_by_key()
    missing = {
        model_key
        for choice in PRETRAINED_BLOCK_PLAN
        for model_key in choice.primary_models
        if model_key not in registry
    }
    assert missing == set()


def test_cpu_first_restoration_models_are_selected() -> None:
    deblur = plan_by_block()[BlockKind.DEBLUR]
    assert deblur.primary_models[:2] == ("nafnet_gopro_width32", "nafnet_sidd_width32")
    assert "restormer_motion_deblur" in deblur.primary_models


def test_strict_reference_fusion_does_not_require_a_checkpoint() -> None:
    region = plan_by_block()[BlockKind.REGION_SELECT]
    fusion = plan_by_block()[BlockKind.FUSION]
    assert "specific reference memory" in region.fallback.lower()
    assert "provenance" in fusion.fallback.lower()


def test_import_and_export_are_deterministic() -> None:
    plan = plan_by_block()
    assert plan[BlockKind.IMPORT].primary_models == ()
    assert plan[BlockKind.EXPORT].primary_models == ()
