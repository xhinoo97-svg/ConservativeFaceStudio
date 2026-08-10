from __future__ import annotations

import numpy as np

from app.execution import Workspace
from app.pipeline import BlockKind, default_pipeline
from app.strict_execution import StrictBlockExecutor


def _image() -> np.ndarray:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[16:80, 16:80] = (90, 130, 170)
    return image


def test_all_13_block_kinds_have_concrete_strict_handlers() -> None:
    executor = StrictBlockExecutor(Workspace(primary=_image()))
    pipeline = default_pipeline()

    assert len(pipeline) == 13
    assert {block.kind for block in pipeline} == set(BlockKind)
    assert set(executor._handlers) == set(BlockKind)

    for block in pipeline:
        handler = executor._handlers.get(block.kind)
        assert callable(handler), block.kind
        name = getattr(handler, "__name__", "")
        assert name
        assert "placeholder" not in name.lower()
        assert "todo" not in name.lower()


def test_every_block_has_explicit_dependency_contract() -> None:
    blocks = default_pipeline()
    assert blocks[0].kind is BlockKind.IMPORT
    assert blocks[-1].kind is BlockKind.EXPORT
    seen: set[str] = set()
    for block in blocks:
        assert block.key not in seen
        assert all(dep in seen for dep in block.depends_on)
        seen.add(block.key)


def test_strict_executor_provides_real_inpaint_and_frontalize_fallbacks() -> None:
    executor = StrictBlockExecutor(Workspace(primary=_image()))
    assert executor._handlers[BlockKind.INPAINT].__name__ == "_reference_repair"
    assert executor._handlers[BlockKind.FRONTALIZE].__name__ == "_pose_normalize"
    assert executor._handlers[BlockKind.OCCLUSION_MASK].__name__ == "_strict_occlusion"
    assert executor._handlers[BlockKind.REGION_SELECT].__name__ == "_specific_memory_select"
    assert executor._handlers[BlockKind.UPSCALE].__name__ == "_strict_upscale"
