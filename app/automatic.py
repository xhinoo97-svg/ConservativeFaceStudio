from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.execution import BlockExecutionError, ExecutionResult, Workspace
from app.pipeline import BlockKind
from app.pretrained_face_handlers import install_pretrained_face_handlers
from app.pretrained_restoration_handlers import install_pretrained_restoration_handlers
from app.pretrained_semantic_handlers import install_pretrained_semantic_handlers
from app.pretrained_values import RESTORATION_SAFETY_DEFAULTS
from app.strict_execution import StrictBlockExecutor
from app.validation import evaluate_identity_guardrail


@dataclass(frozen=True)
class AutomaticRunResult:
    final_image: Path
    provenance: Path | None
    blocks_zip: Path
    results: tuple[ExecutionResult, ...]


class AutomaticPipelineRunner:
    """Esegue l'intera pipeline senza conferme intermedie, mantenendo audit e strict mode."""

    def __init__(self, workspace: Workspace) -> None:
        self.executor = StrictBlockExecutor(workspace)
        core_paths = workspace.metadata.get("core_model_paths")
        if isinstance(core_paths, dict):
            install_pretrained_face_handlers(self.executor, core_paths)
            install_pretrained_restoration_handlers(self.executor, core_paths)
            install_pretrained_semantic_handlers(self.executor, core_paths)
        self.on_progress: Callable[[int, str], None] | None = None
        self._original_anchor = workspace.copy_primary()

    def _emit_progress(self, index: int, name: str) -> None:
        callback = self.on_progress
        if callback is not None:
            callback(int(index), str(name))

    def _apply_guardrail(self, block, before: np.ndarray, result: ExecutionResult) -> ExecutionResult:
        if block.kind in {BlockKind.IMPORT, BlockKind.EXPORT, BlockKind.IDENTITY_CHECK}:
            return result
        anchors = list(self.executor.workspace.references) or [self._original_anchor]
        decision = evaluate_identity_guardrail(
            before,
            result.image,
            anchors,
            max_drop=RESTORATION_SAFETY_DEFAULTS.identity_max_drop,
            absolute_minimum=0.20,
            minimum_retention=RESTORATION_SAFETY_DEFAULTS.identity_minimum_retention,
        )
        details = dict(result.details)
        details["identity_guardrail"] = {
            "accepted": decision.accepted,
            "score_before": decision.score_before,
            "score_after": decision.score_after,
            "score_drop": decision.score_drop,
            "retention_ratio": decision.retention_ratio,
            "minimum_retention": decision.minimum_retention,
            "engine": decision.engine,
            "reason": decision.reason,
        }
        if decision.accepted:
            if self.executor.project.operations:
                self.executor.project.operations[-1].parameters["identity_guardrail"] = details["identity_guardrail"]
            return ExecutionResult(result.block, result.image, details)

        if not np.array_equal(before, result.image):
            restored = self.executor.undo()
        else:
            restored = before.copy()
            self.executor.workspace.primary = restored.copy()
        details["rolled_back"] = True
        details["rollback_reason"] = decision.reason
        details.pop("snapshot_sha256", None)
        replacement = self.executor.block_artifacts.replace_last(restored, details)
        details["snapshot_sha256"] = replacement.sha256
        if self.executor.project.operations:
            self.executor.project.operations[-1].parameters.update({
                "identity_guardrail": details["identity_guardrail"],
                "rolled_back": True,
                "rollback_reason": decision.reason,
                "snapshot_sha256": replacement.sha256,
            })
        return ExecutionResult(result.block, restored, details)

    def run(self, output: str | Path, *, deblur: dict[str, Any] | None = None, upscale: int = 2, identity_minimum: float = 0.35) -> AutomaticRunResult:
        output_path = Path(output)
        blocks = self.executor.pipeline.blocks
        results: list[ExecutionResult] = []
        for index, block in enumerate(blocks, start=1):
            self._emit_progress(index - 1, f"Avvio: {block.title}")
            if block.kind is BlockKind.EXPORT:
                result = self.executor.execute(block, path=output_path, blocks_zip=output_path.with_suffix(output_path.suffix + ".blocks.zip"))
                results.append(result)
                self._emit_progress(index, block.title)
                provenance = result.details.get("provenance_path")
                return AutomaticRunResult(Path(result.details["path"]), Path(provenance) if provenance else None, Path(result.details["blocks_zip"]), tuple(results))

            parameters: dict[str, Any] = {}
            if block.kind is BlockKind.DEBLUR:
                parameters.update(deblur or {})
            elif block.kind is BlockKind.UPSCALE:
                parameters["scale"] = upscale
            elif block.kind is BlockKind.IDENTITY_CHECK:
                parameters["minimum"] = identity_minimum

            reason = self._skip_reason(block.kind)
            if reason is not None:
                results.append(self.executor.record_skipped(block, reason))
                self._emit_progress(index, f"{block.title} — saltato")
                continue

            try:
                before = self.executor.workspace.copy_primary()
                raw = self.executor.execute(block, **parameters)
                result = self._apply_guardrail(block, before, raw)
                results.append(result)
                self._emit_progress(index, block.title + (" — rollback" if result.details.get("rolled_back") else ""))
            except BlockExecutionError as exc:
                if block.kind in {BlockKind.IMPORT, BlockKind.IDENTITY_CHECK}:
                    raise
                results.append(self.executor.record_skipped(block, str(exc)))
                self._emit_progress(index, f"{block.title} — saltato")
            except ValueError as exc:
                results.append(self.executor.record_skipped(block, str(exc)))
                self._emit_progress(index, f"{block.title} — saltato")
        raise RuntimeError("Pipeline terminata senza blocco export")

    def _skip_reason(self, kind: BlockKind) -> str | None:
        has_references = bool(self.executor.workspace.references)
        if kind in {BlockKind.ALIGN, BlockKind.REGION_SELECT, BlockKind.INPAINT, BlockKind.FUSION} and not has_references:
            return "Nessuna fotografia di riferimento disponibile"
        return None
