from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
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
from app.validation import GuardrailDecision, evaluate_identity_guardrail


REFERENCE_PHOTOMETRY_ENHANCE_ABSTENTION = (
    "ENHANCE_ABSTAIN_PRESERVE_OBSERVED_REFERENCE_PHOTOMETRY"
)


@dataclass(frozen=True)
class AutomaticRunResult:
    final_image: Path
    provenance: Path | None
    blocks_zip: Path
    results: tuple[ExecutionResult, ...]


class AutomaticRunCancelled(RuntimeError):
    """Cooperative cancellation observed at a safe block boundary."""

    def __init__(
        self,
        next_block_index: int,
        next_block_key: str,
        completed_results: tuple[ExecutionResult, ...],
    ) -> None:
        super().__init__(
            f"Pipeline annullata prima del blocco {next_block_index}: {next_block_key}"
        )
        self.next_block_index = int(next_block_index)
        self.next_block_key = str(next_block_key)
        self.completed_results = tuple(completed_results)


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
        self._model_paths = model_paths
        self._observed_sources = observed_sources

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
        self.on_block_completed: Callable[[int, str, str, np.ndarray, dict[str, Any]], None] | None = None
        self.should_cancel: Callable[[], bool] | None = None
        self._original_anchor = observed_sources[0].copy()

    def _run_preflight_after_import(self) -> None:
        """Run analysis/restoration only after Block 01 has recorded immutable MAIN."""
        workspace = self.executor.workspace
        if not self._model_paths or bool(workspace.metadata.get("preflight_completed", False)):
            return
        before = workspace.copy_primary()
        try:
            preflight = preprocess_and_select_front_base(workspace, self._model_paths)
            anchor_decision = restore_imported_primary_for_same_canvas(workspace, self._observed_sources)
            apply_observed_restoration_policy(workspace, self._observed_sources)
            changed = np.any(workspace.primary != before, axis=2)
            workspace.metadata["preflight_completed"] = True
            workspace.metadata["preflight_main_changed_pixels"] = int(np.count_nonzero(changed))
            workspace.metadata["preflight_main_mae"] = float(
                np.mean(np.abs(workspace.primary.astype(np.int16) - before.astype(np.int16)))
            )
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
            workspace.primary = before
            workspace.metadata["preflight_completed"] = False
            workspace.metadata["preflight_error"] = str(exc)

    def _emit_progress(self, index: int, name: str) -> None:
        callback = self.on_progress
        if callback is not None:
            callback(int(index), str(name))

    def _emit_block_completed(self, index: int, block, result: ExecutionResult) -> None:
        callback = self.on_block_completed
        if callback is None:
            return
        status = str(result.details.get("status", ""))
        if status not in {"PASS", "ROLLBACK", "ABSTAIN", "SKIPPED", "UNRESOLVED"}:
            status = "ROLLBACK" if result.details.get("rolled_back") else ("SKIPPED" if result.details.get("skipped") else "PASS")
        callback(int(index), str(block.title), status, result.image.copy(), dict(result.details))

    def _record_final_identity_rollback(self, block, reason: str) -> ExecutionResult:
        """Fail closed to immutable MAIN without reporting rejection as a crash."""
        workspace = self.executor.workspace
        rejected = workspace.copy_primary()
        restored = self._original_anchor.copy()
        changed = (
            np.any(rejected != restored, axis=2)
            if rejected.shape == restored.shape
            else np.ones(restored.shape[:2], dtype=bool)
        )
        unresolved = self._binary_mask(
            workspace.metadata.get("inpaint_unresolved_mask"), restored.shape[:2]
        )
        unresolved |= changed

        workspace.primary = restored.copy()
        workspace.provenance_map = np.zeros(restored.shape[:2], dtype=np.uint16)
        workspace.metadata["inpaint_unresolved_mask"] = np.where(
            unresolved, 255, 0
        ).astype(np.uint8)
        workspace.metadata.update({
            "runtime_success": True,
            "identity_safe": True,
            "restoration_effective": False,
            "abstained": False,
            "unresolved": bool(np.any(unresolved)),
            "hard_guardrail_pass": True,
            "final_identity_status": "ROLLBACK",
            "final_identity_failure_reason": str(reason),
            "identity_fail_closed_source": "IMMUTABLE_MAIN",
            "zero_recovery_is_restoration_pass": False,
        })
        self.executor.history.restore_discarding_later(restored, "identity-rollback")

        result = self.executor.record_skipped(block, str(reason))
        details = dict(result.details)
        details.update({
            "status": "ROLLBACK",
            "skipped": False,
            "rolled_back": True,
            "runtime_success": True,
            "identity_safe": True,
            "restoration_effective": False,
            "abstained": False,
            "unresolved": bool(np.any(unresolved)),
            "hard_guardrail_pass": True,
            "rollback_source": "IMMUTABLE_MAIN",
            "rollback_reason": str(reason),
            "rejected_candidate_changed_pixels": int(np.count_nonzero(changed)),
            "unresolved_pixels": int(np.count_nonzero(unresolved)),
            "wrong_person_final_pixels": 0,
            "zero_recovery_is_restoration_pass": False,
        })
        replacement = self.executor.block_artifacts.replace_last(restored, details)
        details["snapshot_sha256"] = replacement.sha256
        if self.executor.project.operations:
            self.executor.project.operations[-1].parameters.update(details)
        return ExecutionResult(block.key, restored, details)

    @staticmethod
    def _binary_mask(value: Any, shape: tuple[int, int]) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            return np.zeros(shape, dtype=bool)
        item = np.asarray(value)
        if item.ndim == 3:
            item = cv2.cvtColor(item.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        if item.shape != shape:
            return np.zeros(shape, dtype=bool)
        return item > 0

    def _authorized_observed_repair_domain(self, shape: tuple[int, int]) -> np.ndarray:
        workspace = self.executor.workspace
        target = np.zeros(shape, dtype=bool)
        frozen = workspace.metadata.get("preflight_original_occlusion_masks")
        if isinstance(frozen, list) and frozen:
            target |= self._binary_mask(np.asarray(frozen[0]), shape)
        if isinstance(workspace.occlusion_masks, list) and workspace.occlusion_masks:
            target |= self._binary_mask(np.asarray(workspace.occlusion_masks[0]), shape)
        for key in ("reference_consensus_occlusion", "inpaint_target_mask"):
            target |= self._binary_mask(workspace.metadata.get(key), shape)
        if np.any(target):
            # Photometric feathering may legitimately touch only a narrow target edge.
            target = cv2.dilate(
                target.astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ) > 0
        return target

    def _trusted_reference_source_codes(self) -> set[int]:
        """Return imported reference provenance codes already verified upstream.

        Partial same-canvas geometry is a local trust statement, not a global face
        identity anchor.  It is nevertheless sufficient to preserve an exact observed
        donor transfer inside the verified damage ROI.  Full identity-verified donors
        are accepted here as well.
        """
        workspace = self.executor.workspace
        originals_raw = workspace.metadata.get("aligned_reference_original_source_indices")
        if not isinstance(originals_raw, list):
            return set()
        try:
            originals = [max(1, int(value)) for value in originals_raw]
        except (TypeError, ValueError):
            return set()

        count = len(originals)
        identity = workspace.metadata.get("aligned_reference_identity_verified")
        partial = workspace.metadata.get("aligned_reference_partial_geometry_verified")
        identity_flags = [False] * count
        partial_flags = [False] * count
        if isinstance(identity, list) and len(identity) == count:
            identity_flags = [bool(value) for value in identity]
        if isinstance(partial, list) and len(partial) == count:
            partial_flags = [bool(value) for value in partial]

        trusted = {
            originals[index]
            for index in range(count)
            if identity_flags[index] or partial_flags[index]
        }

        # Same-canvas verification records runtime indices explicitly. Resolve them to
        # the already aligned original-source list when possible.
        runtime_raw = workspace.metadata.get("aligned_reference_source_indices")
        runtime = [int(value) for value in runtime_raw] if isinstance(runtime_raw, list) and len(runtime_raw) == count else list(range(count))
        runtime_to_original = {runtime[index]: originals[index] for index in range(count)}
        for key in ("verified_same_canvas_alignment", "same_canvas_partial_alignment_diagnostics"):
            diagnostics = workspace.metadata.get(key)
            if not isinstance(diagnostics, list):
                continue
            for item in diagnostics:
                if not isinstance(item, dict) or item.get("runtime_reference_index") is None:
                    continue
                method = str(item.get("method", ""))
                if method not in {"verified-same-canvas-observed", "verified-same-canvas-partial"}:
                    continue
                original = runtime_to_original.get(int(item["runtime_reference_index"]))
                if original is not None:
                    trusted.add(int(original))
        return trusted

    def _trusted_observed_reference_change(
        self,
        block,
        before: np.ndarray,
        candidate: np.ndarray,
    ) -> tuple[bool, dict[str, Any]]:
        """Recognise exact observed donor transfers that must not be judged as global sparse faces.

        A partial reference is not a valid whole-face identity anchor. If blocks 7-9
        changed only an authorised damaged ROI and every changed pixel is provenance-
        backed by an already verified observed reference, the transfer has already
        passed the stronger source/geometry contract. Global histogram/SFace comparison
        against a mostly-black partial sheet would be the wrong validation class.
        """
        if block.kind not in {BlockKind.REGION_SELECT, BlockKind.INPAINT, BlockKind.FUSION}:
            return False, {"reason": "block_not_reference_repair"}
        if candidate.shape != before.shape:
            return False, {"reason": "shape_changed"}

        changed = np.any(candidate != before, axis=2)
        changed_pixels = int(np.count_nonzero(changed))
        if changed_pixels == 0:
            return False, {"reason": "no_pixel_change"}

        workspace = self.executor.workspace
        provenance = workspace.provenance_map
        if not isinstance(provenance, np.ndarray) or provenance.shape != changed.shape:
            return False, {"reason": "missing_provenance", "changed_pixels": changed_pixels}
        provenance = provenance.astype(np.uint16, copy=False)

        observed_reference = changed & (provenance > 0) & (provenance < np.uint16(65534))
        if int(np.count_nonzero(observed_reference)) != changed_pixels:
            return False, {
                "reason": "change_not_fully_observed_reference",
                "changed_pixels": changed_pixels,
                "observed_reference_changed_pixels": int(np.count_nonzero(observed_reference)),
            }

        trusted_codes = self._trusted_reference_source_codes()
        changed_codes = {int(value) for value in np.unique(provenance[changed]) if 0 < int(value) < 65534}
        if not changed_codes or not changed_codes.issubset(trusted_codes):
            return False, {
                "reason": "reference_source_not_verified",
                "changed_source_codes": sorted(changed_codes),
                "trusted_source_codes": sorted(trusted_codes),
            }

        authorized = self._authorized_observed_repair_domain(changed.shape)
        outside = changed & ~authorized
        if np.any(outside):
            return False, {
                "reason": "observed_transfer_outside_authorized_damage",
                "changed_pixels": changed_pixels,
                "outside_authorized_pixels": int(np.count_nonzero(outside)),
            }

        return True, {
            "reason": "trusted_observed_reference_transfer",
            "changed_pixels": changed_pixels,
            "source_codes": sorted(changed_codes),
            "trusted_source_codes": sorted(trusted_codes),
            "outside_authorized_pixels": 0,
        }

    def _global_identity_anchors(self) -> list[np.ndarray]:
        """Exclude sparse/partial sheets from whole-face identity scoring."""
        workspace = self.executor.workspace
        refs = list(workspace.references)
        candidates = workspace.metadata.get("preflight_candidates")
        if not isinstance(candidates, list):
            return [self._original_anchor]

        accepted_sources: set[int] = set()
        for item in candidates:
            if not isinstance(item, dict) or not bool(item.get("accepted_identity", False)):
                continue
            try:
                source_index = int(item.get("source_index"))
            except (TypeError, ValueError):
                continue
            if source_index > 0:
                accepted_sources.add(source_index)

        order_raw = workspace.metadata.get("runtime_source_order")
        order = [int(value) for value in order_raw] if isinstance(order_raw, list) and len(order_raw) == len(refs) + 1 else list(range(len(refs) + 1))
        full_face: list[np.ndarray] = []
        for runtime_index, reference in enumerate(refs):
            original_source = order[runtime_index + 1] if runtime_index + 1 < len(order) else runtime_index + 1
            if original_source in accepted_sources:
                full_face.append(reference)
        return full_face or [self._original_anchor]

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

        if block.kind is BlockKind.UPSCALE:
            scale = int(result.details.get("scale", 1))
            expected_shape = (before.shape[0] * scale, before.shape[1] * scale)
            geometry_valid = scale >= 1 and result.image.shape[:2] == expected_shape
            details = dict(result.details)
            details["identity_guardrail"] = {
                "accepted": bool(geometry_valid),
                "engine": "deterministic-transform-consistency",
                "reason": "accepted deterministic scale transform" if geometry_valid else "invalid deterministic scale geometry",
                "scale": scale,
                "source_dimensions": [int(before.shape[1]), int(before.shape[0])],
                "target_dimensions": [int(result.image.shape[1]), int(result.image.shape[0])],
            }
            if geometry_valid:
                if self.executor.project.operations:
                    self.executor.project.operations[-1].parameters["identity_guardrail"] = details["identity_guardrail"]
                return ExecutionResult(result.block, result.image, details)
            raise BlockExecutionError("Upscale deterministico con geometria incoerente")

        trusted_observed, observed_details = self._trusted_observed_reference_change(block, before, result.image)
        if trusted_observed:
            decision = GuardrailDecision(
                True,
                1.0,
                1.0,
                0.0,
                "trusted-observed-reference-provenance",
                "accepted: exact observed reference transfer already passed source/geometry gate",
                1.0,
                RESTORATION_SAFETY_DEFAULTS.identity_minimum_retention,
            )
        else:
            anchors = self._global_identity_anchors()
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
            "trusted_observed_reference_transfer": bool(trusted_observed),
            "trusted_observed_details": observed_details,
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
            if self.should_cancel is not None and bool(self.should_cancel()):
                self.executor.workspace.metadata.update(
                    {
                        "runtime_success": False,
                        "runtime_cancelled": True,
                        "cancelled_before_block_index": int(index),
                        "cancelled_before_block_key": str(block.key),
                        "last_accepted_block_key": (
                            results[-1].block if results else None
                        ),
                    }
                )
                raise AutomaticRunCancelled(
                    index,
                    block.key,
                    tuple(results),
                )
            if block.kind is BlockKind.DEBLUR:
                self._run_preflight_after_import()
            self._emit_progress(index - 1, f"Avvio: {block.title}")
            if block.kind is BlockKind.EXPORT:
                result = self.executor.execute(block, path=output_path, blocks_zip=output_path.with_suffix(output_path.suffix + ".blocks.zip"))
                results.append(result)
                self._emit_block_completed(index, block, result)
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
                parameters["maximum_occlusion_fraction"] = 1.0
            elif block.kind is BlockKind.FUSION:
                parameters["maximum_occlusion_fraction"] = 1.0
            elif block.kind is BlockKind.UPSCALE:
                parameters["scale"] = upscale
            elif block.kind is BlockKind.IDENTITY_CHECK:
                parameters["minimum"] = identity_minimum

            reason = self._skip_reason(block.kind)
            if reason is not None:
                decision_metadata: dict[str, Any] = {}
                if (
                    block.kind is BlockKind.ENHANCE
                    and reason == REFERENCE_PHOTOMETRY_ENHANCE_ABSTENTION
                ):
                    decision_metadata = {
                        "status": "ABSTAIN",
                        "decision": "ABSTAIN",
                        "abstained": True,
                        "restoration_effective": False,
                        "restoration_pass": False,
                        "zero_recovery_is_restoration_pass": False,
                        "engine": "automatic-reference-photometry-preserve",
                        "reference_evidence_preserved": True,
                    }
                skipped = self.executor.record_skipped(
                    block,
                    reason,
                    **decision_metadata,
                )
                results.append(skipped)
                self._emit_block_completed(index, block, skipped)
                self._emit_progress(index, f"{block.title} — saltato")
                continue

            try:
                before = self.executor.workspace.copy_primary()
                state_before = self._snapshot_guardrail_state()
                raw = self.executor.execute(block, **parameters)
                result = self._apply_guardrail(block, before, raw, state_before)
                results.append(result)
                self._emit_block_completed(index, block, result)
                self._emit_progress(index, block.title + (" — rollback" if result.details.get("rolled_back") else ""))
            except BlockExecutionError as exc:
                if block.kind is BlockKind.IMPORT:
                    raise
                if block.kind is BlockKind.IDENTITY_CHECK:
                    result = self._record_final_identity_rollback(block, str(exc))
                    results.append(result)
                    self._emit_block_completed(index, block, result)
                    self._emit_progress(index, f"{block.title} — rollback")
                    continue
                skipped = self.executor.record_skipped(block, str(exc))
                results.append(skipped)
                self._emit_block_completed(index, block, skipped)
                self._emit_progress(index, f"{block.title} — saltato")
            except ValueError as exc:
                skipped = self.executor.record_skipped(block, str(exc))
                results.append(skipped)
                self._emit_block_completed(index, block, skipped)
                self._emit_progress(index, f"{block.title} — saltato")
        raise RuntimeError("Pipeline terminata senza blocco export")

    def _skip_reason(self, kind: BlockKind) -> str | None:
        workspace = self.executor.workspace
        has_references = bool(workspace.references)
        if kind is BlockKind.ENHANCE and has_references:
            return REFERENCE_PHOTOMETRY_ENHANCE_ABSTENTION
        if kind in {BlockKind.ALIGN, BlockKind.REGION_SELECT, BlockKind.FUSION} and not has_references:
            return "Nessuna fotografia di riferimento disponibile"
        if kind is BlockKind.FRONTALIZE:
            target = workspace.metadata.get("inpaint_target_mask")
            has_target = isinstance(target, np.ndarray) and bool(np.any(target > 0))
            if not has_target and isinstance(workspace.occlusion_masks, list):
                has_target = any(isinstance(mask, np.ndarray) and bool(np.any(mask > 0)) for mask in workspace.occlusion_masks)
            if has_target:
                return "Restauro conservativo: preservata la geometria originale della fotografia primaria"
        return None
