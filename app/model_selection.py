from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.model_registry import inspect_model, registry_by_key
from app.pipeline import BlockKind
from app.pretrained_plan import plan_by_block


MODEL_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "opencv_sface": ("opencv_yunet",),
}


@dataclass(frozen=True)
class ModelSelection:
    block: BlockKind
    model_key: str | None
    model_path: Path | None
    uses_pretrained: bool
    fallback: str
    execution_policy: str
    reason: str


def _installed(key: str, root: str | Path) -> bool:
    registry = registry_by_key()
    status = inspect_model(registry[key], root)
    if not bool(status["exists"]):
        return False
    return all(_installed(dependency, root) for dependency in MODEL_DEPENDENCIES.get(key, ()))


def select_model_for_block(block: BlockKind, root: str | Path = ".") -> ModelSelection:
    """Select the highest-priority installed pretrained model for a block.

    Selection never downloads anything and never makes a missing optional model a
    pipeline error. Model order is defined centrally in ``app.pretrained_plan``.
    Dependencies such as SFace -> YuNet must also be installed.
    """
    choice = plan_by_block()[block]
    registry = registry_by_key()
    for key in choice.primary_models:
        if _installed(key, root):
            status = inspect_model(registry[key], root)
            return ModelSelection(
                block=block,
                model_key=key,
                model_path=Path(str(status["path"])),
                uses_pretrained=True,
                fallback=choice.fallback,
                execution_policy=choice.execution_policy,
                reason=choice.reason,
            )
    return ModelSelection(
        block=block,
        model_key=None,
        model_path=None,
        uses_pretrained=False,
        fallback=choice.fallback,
        execution_policy=choice.execution_policy,
        reason=choice.reason,
    )


def installed_plan(root: str | Path = ".") -> tuple[ModelSelection, ...]:
    return tuple(select_model_for_block(block, root) for block in BlockKind)
