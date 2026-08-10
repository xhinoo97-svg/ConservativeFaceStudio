from __future__ import annotations

import inspect

import numpy as np

from app.execution import Workspace
from app.pipeline import BlockKind
from app.pretrained_plan import plan_by_block
from app.strict_execution import StrictBlockExecutor


CORE = (
    BlockKind.ALIGN,
    BlockKind.OCCLUSION_MASK,
    BlockKind.REGION_SELECT,
    BlockKind.INPAINT,
    BlockKind.FUSION,
)


def _workspace() -> Workspace:
    primary = np.full((96, 96, 3), 100, dtype=np.uint8)
    refs = [primary.copy() for _ in range(9)]
    return Workspace(primary=primary, references=refs)


def test_core_5_9_have_real_handlers_and_explicit_model_fallback_contracts() -> None:
    executor = StrictBlockExecutor(_workspace())
    plans = plan_by_block()

    for kind in CORE:
        handler = executor._handlers.get(kind)
        assert callable(handler), kind
        name = getattr(handler, "__name__", "").lower()
        assert name and "placeholder" not in name and "todo" not in name

        plan = plans[kind]
        assert plan.execution_policy
        assert plan.fallback
        assert plan.reason
        # ALIGN and REGION_SELECT/FUSION use pretrained geometry/semantic/identity
        # assistance; INPAINT has LaMa as residual fallback; OCCLUSION has face parsing.
        assert plan.primary_models, kind


def test_core_5_9_runtime_is_not_hard_limited_to_five_references() -> None:
    executor = StrictBlockExecutor(_workspace())
    region = executor._handlers[BlockKind.REGION_SELECT]
    source = inspect.getsource(region)
    # The installed multi-reference policy makes an unspecified top_k dynamic. A stale
    # explicit top_k=2/5 here would silently discard references 6..9.
    assert "top_k=2" not in source
    assert "top_k = 2" not in source
    assert "top_k=5" not in source
    assert "top_k = 5" not in source


def test_core_5_9_conservative_modules_are_importable() -> None:
    from app import component_alignment  # noqa: F401
    from app import component_bank  # noqa: F401
    from app import reference_guided_seed_policy  # noqa: F401
    from app import reference_memory  # noqa: F401
    from app import observed_target_repair_runtime  # noqa: F401
    from app import observed_target_fusion_runtime  # noqa: F401
    from app import explicit_damage_domain_policy  # noqa: F401
    from app import same_canvas_seed_support_policy  # noqa: F401
