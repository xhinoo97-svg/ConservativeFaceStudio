from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.execution import BlockExecutionError, BlockExecutor, ExecutionResult, Workspace
from app.pipeline import BlockKind


@dataclass(frozen=True)
class AutomaticRunResult:
    final_image: Path
    provenance: Path | None
    blocks_zip: Path
    results: tuple[ExecutionResult, ...]


class AutomaticPipelineRunner:
    """Esegue l'intera pipeline senza conferme intermedie, mantenendo audit e strict mode."""

    def __init__(self, workspace: Workspace) -> None:
        self.executor = BlockExecutor(workspace)

    def run(
        self,
        output: str | Path,
        *,
        deblur: dict[str, Any] | None = None,
        upscale: int = 2,
        identity_minimum: float = 0.35,
    ) -> AutomaticRunResult:
        output_path = Path(output)
        blocks = self.executor.pipeline.blocks
        results: list[ExecutionResult] = []

        for block in blocks:
            if block.kind is BlockKind.EXPORT:
                result = self.executor.execute(
                    block,
                    path=output_path,
                    blocks_zip=output_path.with_suffix(output_path.suffix + ".blocks.zip"),
                )
                results.append(result)
                provenance = result.details.get("provenance_path")
                return AutomaticRunResult(
                    final_image=Path(result.details["path"]),
                    provenance=Path(provenance) if provenance else None,
                    blocks_zip=Path(result.details["blocks_zip"]),
                    results=tuple(results),
                )

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
                continue

            try:
                results.append(self.executor.execute(block, **parameters))
            except BlockExecutionError as exc:
                if block.kind in {BlockKind.IMPORT, BlockKind.IDENTITY_CHECK}:
                    raise
                results.append(self.executor.record_skipped(block, str(exc)))
            except ValueError as exc:
                results.append(self.executor.record_skipped(block, str(exc)))

        raise RuntimeError("Pipeline terminata senza blocco export")

    def _skip_reason(self, kind: BlockKind) -> str | None:
        if kind in {BlockKind.LANDMARKS, BlockKind.INPAINT, BlockKind.FRONTALIZE}:
            return "Modulo opzionale non installato o disattivato in strict mode"
        has_references = bool(self.executor.workspace.references)
        if kind in {BlockKind.ALIGN, BlockKind.REGION_SELECT, BlockKind.FUSION} and not has_references:
            return "Nessuna fotografia di riferimento disponibile"
        return None
