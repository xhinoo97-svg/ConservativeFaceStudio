from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.case_aware_runtime import install_case_aware_runtime
from app.conservative_observed_runtime import install_conservative_observed_runtime
from app.execution import BlockExecutionError, ExecutionResult, Workspace
from app.observed_restoration_policy import apply_observed_restoration_policy
from app.observed_target_repair_runtime import install_observed_target_repair_runtime
from app.partial_reference_runtime import install_partial_reference_runtime
from app.pipeline import BlockKind
from app.preflight import preprocess_and_select_front_base
from app.pretrained_face_handlers import install_pretrained_face_handlers
from app.pretrained_inpaint_handler import install_verified_inpainting_handler
from app.pretrained_restoration_handlers import install_pretrained_restoration_handlers
from app.pretrained_semantic_handlers import install_pretrained_semantic_handlers
from app.pretrained_values import RESTORATION_SAFETY_DEFAULTS
from app.primary_anchor_policy import restore_imported_primary_for_same_canvas
from app.same_canvas_repair_runtime import install_same_canvas_repair_runtime
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

    _GUARDRAIL_METADATA_KEYS = (
        "specific_reference_confidence",
        "specific_reference_memory",
        "inpaint_target_mask",
        "inpaint_observed_mask",
        "inpaint_symmetry_mask",
        "inpaint_generated_mask",
        "inpaint_unresolved_mask",
        "primary_landmarks5",
        "aligned_reference_support_masks",
        "component_reference_bank",
        "component_alignment_diagnostics",
        "restoration_case",
        "restoration_case_assessment",
    )

    def __init__(self, workspace: Workspace) -> None:
        core_paths = workspace.metadata.get("core_model_paths")
        model_paths = core_paths if isinstance(core_paths, dict) else {}
        observed_sources = [workspace.primary.copy(), *[item.copy() for item in workspace.references]]

        if model_paths and not bool(workspace.metadata.get("preflight_completed", False)):
            try:
                preflight = preprocess_and_select_front_base(workspace, model_paths)
                anchor_decision = restore_imported_primary_for_same_canvas(workspace, observed_sources)
                apply_observed_restoration_policy(workspace, observed_sources)
                workspace.metadata["preflight_completed"] = True
                workspace.metadata["preflight_selected_source_index"] = int(preflight.selected_source_index)
                workspace.metadata["preflight_runtime_primary_source_index"] = int(
                    workspace.metadata.get("selected_primary_original_source_index", preflight.selected_source_index)
                )
                workspace.metadata["preflight_identity_cluster_size"] = int(preflight.identity_cluster_size)
                workspace.metadata["preflight_reason"] = str(preflight.reason)
                workspace.metadata["preflight_candidate_count"] = len(preflight.candidates)
                workspace.metadata["primary_anchor_policy"] = {
                    "applied": bool(anchor_decision.applied),
                    "reason": str(anchor_decision.reason),
                    "matched_reference_count": int(anchor_decision.matched_reference_count),
                    "original_selected_source_index": int(anchor_decision.original_selected_source_index),
                }
            except Exception as exc:
                workspace.metadata["preflight_completed"] = False
                workspace.metadata["preflight_error"] = str(exc)

        self.executor = StrictBlockExecutor(workspace)
        if model_paths:
            install_pretrained_face_handlers(self.executor, model_paths)
            install_pretrained_restoration_handlers(self.executor, model_paths)
            install_pretrained_semantic_handlers(self.executor, model_paths)
        install_verified_inpainting_handler(self.executor, model_paths)
        install_partial_reference_runtime(self.executor)
        install_case_aware_runtime(self.executor, model_paths)
        install_conservative_observed_runtime(self.executor)
        install_same_canvas_repair_runtime(self.executor)
        install_observed_target_repair_runtime(self.executor)
        self.on_progress: Callable[[int, str], None] | None = None
        self._original_anchor = workspace.copy_primary()

    def _emit_progress(self, index: int, name: str) -> None:
        callback = self.on_progress
        if callback is not None:
            callback(int(index), str(name))

    def _snapshot_guardrail_state(self) -> dict[str, Any]:
        workspace = self.executor.workspace
        provenance = None if workspace.provenance_map is None else workspace.provenance_map.copy()
        metadata: dict[str, tuple[bool, Any]] = {}
        for key in self._GUARDRAIL_METADATA_KEYS:
            if key not in workspace.metadata:
                metadata[key] = (False, None)
                continue
            value = workspace.metadata[key]
            if isinstance(value, np.ndarray):
                stored = value.copy()
            else:
                stored = copy.deepcopy(value)
            metadata[key] = (True, stored)
        return {"provenance_map": provenance, "metadata": metadata}

    def _restore_guardrail_state(self, state: dict[str, Any]) -> None:
        workspace = self.executor.workspace
        provenance = state.get("provenance_map")
        workspace.provenance_map = None if provenance is None else np.asarray(provenance).copy()
        metadata = state.get("metadata", {})
        if not isinstance(metadata, dict):
            return
        for key, item in metadata.items():
            present, value = item
            if not present:
                workspace.metadata.pop(key, None)
            elif isinstance(value, np.ndarray):
                workspace.metadata[key] = value.copy()
            else:
                workspace.metadata[key] = copy.deepcopy(value)

    def _apply_guardrail(
        self,
        block,
        before: np.ndarray,
        result: ExecutionResult,
        state_before: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        if block.kind in {BlockKind.IMPORT, BlockKind.EXPORT, BlockKind.IDENTITY_CHECK}:
            return result
        anchors = list(self.executor.workspace.references) or [self._original_anchor]
        identity_backend = self.executor.workspace.metadata.get("_identity_backend")
        decision = evaluate_identity_guardrail(
            before,
            result.image,
            anchors,
            max_drop=RESTORATION_SAFETY_DEFAULTS.identity_max_drop,
            absolute_minimum=0.20,
            minimum_retention=RESTORATION_SAFETY_DEFAULTS.identity_minimum_retention,
            backend=identity_backend,
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
            restored = self.executor.history.rollback_discard_current()
            self.executor.workspace.primary = restored.copy()
        else:
            restored = before.copy()
            self.executor.workspace.primary = restored.copy()
        if state_before is not None:
            self._restore_guardrail_state(state_before)
        details["rolled_back"] = True
        details["rollback_reason"] = decision.reason
        details["workspace_state_restored"] = state_before is not None
        details["rejected_history_discarded"] = True
        details.pop("snapshot_sha256", None)
        replacement = self.executor.block_artifacts.replace_last(restored, details)
        details["snapshot_sha256"] = replacement.sha256
        if self.executor.project.operations:
            self.executor.project.operations[-1].parameters.update({
                "identity_guardrail": details["identity_guardrail"],
                "rolled_back": True,
                "rollback_reason": decision.reason,
                "workspace_state_restored": details["workspace_state_restored"],
                "rejected_history_discarded": True,
                "snapshot_sha256": replacement.sha256,
            })
        return ExecutionResult(result.block, restored, details)

    def run(self, output: str | Path, *, deblur: dict[str, Any] | None = None, upscale: int = 2, identity_minimum: float = 0.363) -> AutomaticRunResult:
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
            elif block.kind is BlockKind.ENHANCE:
                parameters["blend"] = 0.0
            elif block.kind is BlockKind.INPAINT:
                parameters["allow_verified_generative"] = True
                parameters["maximum_generated_face_fraction"] = 0.015
                parameters["maximum_generated_target_fraction"] = 0.25
                parameters["maximum_symmetry_face_fraction"] = 0.08
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
                state_before = self._snapshot_guardrail_state()
                raw = self.executor.execute(block, **parameters)
                result = self._apply_guardrail(block, before, raw, state_before)
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
        if kind in {BlockKind.ALIGN, BlockKind.REGION_SELECT, BlockKind.FUSION} and not has_references:
            return "Nessuna fotografia di riferimento disponibile"
        return None
