from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.execution import ExecutionResult
from app.immutable_input_store import ensure_immutable_input_store
from app.pipeline import BlockKind
from app.severity_aware_deblur_policy import classify_blur

CLEAN = np.uint8(0)
LIGHT = np.uint8(1)
MEDIUM = np.uint8(2)
SEVERE = np.uint8(3)
MISSING = np.uint8(4)
NON_RECOVERABLE_FROM_THIS_IMAGE = np.uint8(5)

_STAGE_STATE_KEYS = (
    "inpaint_target_mask",
    "inpaint_observed_mask",
    "inpaint_symmetry_mask",
    "inpaint_generated_mask",
    "inpaint_unresolved_mask",
    "protected_region_mask",
)


@dataclass(frozen=True)
class StageReport:
    name: str
    requested_pixels: int
    changed_pixels: int
    generated_pixels: int
    unresolved_pixels: int
    accepted: bool
    reason: str


def _binary(value: Any, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        return np.zeros(shape, np.uint8)
    item = np.asarray(value)
    if item.ndim == 3:
        item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        return np.zeros(shape, np.uint8)
    return np.where(item > 0, 255, 0).astype(np.uint8)


def _face_domain(workspace) -> np.ndarray:
    h, w = workspace.primary.shape[:2]
    bbox = workspace.metadata.get("primary_bbox")
    mask = np.zeros((h, w), np.uint8)
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        x, y, bw, bh = (int(v) for v in bbox)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + max(1, bw)), min(h, y + max(1, bh))
        if x2 > x1 and y2 > y1:
            cv2.ellipse(
                mask,
                ((x1 + x2) // 2, (y1 + y2) // 2),
                (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)),
                0,
                0,
                360,
                255,
                -1,
            )
            return mask
    mask[:] = 255
    return mask


def _damage_mask(workspace) -> np.ndarray:
    shape = workspace.primary.shape[:2]
    target = np.zeros(shape, np.uint8)
    frozen = workspace.metadata.get("preflight_original_occlusion_masks")
    if isinstance(frozen, list) and frozen:
        target = cv2.bitwise_or(target, _binary(frozen[0], shape))
    if isinstance(workspace.occlusion_masks, list) and workspace.occlusion_masks:
        target = cv2.bitwise_or(target, _binary(workspace.occlusion_masks[0], shape))
    stored = workspace.metadata.get("inpaint_target_mask")
    target = cv2.bitwise_or(target, _binary(stored, shape))
    return target


def _original_main_for_analysis(workspace) -> np.ndarray:
    """Severity is measured from the imported photograph, never an accepted candidate."""
    try:
        original = ensure_immutable_input_store(workspace).copy_main()
        if original.shape == workspace.primary.shape:
            return original
    except Exception:
        pass
    return workspace.copy_primary()


def build_severity_map(workspace) -> np.ndarray:
    """Build local severity from frozen evidence and immutable MAIN observations."""
    image = _original_main_for_analysis(workspace)
    shape = workspace.primary.shape[:2]
    face = _face_domain(workspace)
    face_pixels = max(1, int(np.count_nonzero(face)))
    damage = cv2.bitwise_and(_damage_mask(workspace), face)
    severity = np.zeros(shape, np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats((damage > 0).astype(np.uint8), 8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        ratio = area / face_pixels
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_frame = x <= 0 or y <= 0 or x + bw >= shape[1] or y + bh >= shape[0]
        level = MISSING if touches_frame and ratio >= 0.05 else (
            LIGHT if ratio <= 0.012 else MEDIUM if ratio <= 0.08 else SEVERE
        )
        severity[labels == label] = level

    reliability_maps = workspace.metadata.get("preflight_detail_reliability_maps")
    reliability = None
    if isinstance(reliability_maps, list) and reliability_maps:
        candidate = np.asarray(reliability_maps[0])
        if candidate.shape == shape:
            reliability = candidate.astype(np.uint8, copy=False)

    blur_info = classify_blur(image)
    workspace.metadata["adaptive_blur_classification"] = dict(blur_info)
    level = str(blur_info.get("level", "none"))
    if reliability is not None and level in {"mild", "medium", "strong"}:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)), (0, 0), 1.2)
        clean = (face > 0) & (damage == 0)
        if level == "mild":
            blur_roi = clean & (reliability < 48) & (lap < 24.0)
            blur_class = LIGHT
        elif level == "medium":
            blur_roi = clean & (reliability < 40) & (lap < 20.0)
            blur_class = MEDIUM
        else:
            blur_roi = clean & (reliability < 32) & (lap < 16.0)
            blur_class = SEVERE
        roi = cv2.morphologyEx(
            blur_roi.astype(np.uint8),
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        ) > 0
        severity[roi & (severity == CLEAN)] = blur_class

    workspace.metadata["adaptive_severity_map"] = severity.copy()
    workspace.metadata["adaptive_severity_counts"] = {
        "clean": int(np.count_nonzero(severity == CLEAN)),
        "light": int(np.count_nonzero(severity == LIGHT)),
        "medium": int(np.count_nonzero(severity == MEDIUM)),
        "severe": int(np.count_nonzero(severity == SEVERE)),
        "missing": int(np.count_nonzero(severity == MISSING)),
        "non_recoverable_from_this_image": int(np.count_nonzero(severity == NON_RECOVERABLE_FROM_THIS_IMAGE)),
    }
    workspace.metadata["adaptive_severity_from_immutable_main"] = True
    return severity


def _snapshot_stage_state(workspace) -> dict[str, Any]:
    state: dict[str, Any] = {
        "primary": workspace.copy_primary(),
        "provenance_map": None if workspace.provenance_map is None else workspace.provenance_map.copy(),
        "metadata": {},
    }
    for key in _STAGE_STATE_KEYS:
        if key not in workspace.metadata:
            state["metadata"][key] = (False, None)
            continue
        value = workspace.metadata[key]
        state["metadata"][key] = (
            True,
            value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value),
        )
    return state


def _restore_stage_state(workspace, state: dict[str, Any]) -> None:
    workspace.primary = np.asarray(state["primary"]).copy()
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


def _remaining_mask(severity: np.ndarray) -> np.ndarray:
    return severity != CLEAN


def install_adaptive_restoration_cascade(executor) -> None:
    """Transactional LIGHT→MEDIUM→SEVERE cascade over only unresolved regions.

    Every non-clean ROI starts at LIGHT.  Only pixels not resolved/accepted there are
    eligible for MEDIUM, and only the remaining residue can reach SEVERE.  Therefore a
    severe sticker still benefits from conservative reference recovery before any
    generative fallback, while an already-fixed eye is protected from later stages.
    """
    original = executor._handlers.get(BlockKind.INPAINT)
    if original is None or getattr(original, "_adaptive_restoration_cascade", False):
        return

    def cascade(block, parameters: dict[str, Any]) -> ExecutionResult:
        workspace = executor.workspace
        severity = build_severity_map(workspace)
        reports: list[StageReport] = []
        final_details: dict[str, Any] = {
            "engine": "adaptive-light-medium-severe",
            "adaptive_cascade": True,
            "transactional_stages": True,
            "protected_regions": True,
            "severity_from_immutable_main": True,
        }

        remaining = _remaining_mask(severity)
        protected = _binary(workspace.metadata.get("protected_region_mask"), remaining.shape) > 0
        remaining &= ~protected
        any_stage = bool(np.any(remaining))

        for stage_name in ("light", "medium", "severe"):
            active = remaining & ~protected
            requested = int(np.count_nonzero(active))
            if requested == 0:
                reports.append(StageReport(stage_name, 0, 0, 0, 0, True, "not_required"))
                continue

            stage_mask = np.where(active, 255, 0).astype(np.uint8)
            state = _snapshot_stage_state(workspace)
            before = workspace.copy_primary()
            provenance_before = None if workspace.provenance_map is None else workspace.provenance_map.copy()
            protected_before = protected.copy()

            workspace.metadata["adaptive_restoration_stage"] = stage_name
            workspace.metadata["adaptive_restoration_stage_mask"] = stage_mask.copy()
            workspace.metadata["protected_region_mask"] = np.where(protected, 255, 0).astype(np.uint8)

            p = dict(parameters)
            p["allow_verified_generative"] = stage_name == "severe"
            if stage_name != "severe":
                p["maximum_generated_face_fraction"] = 0.0
                p["maximum_generated_target_fraction"] = 0.0

            try:
                raw = original(block, p)
            except Exception as exc:
                _restore_stage_state(workspace, state)
                reports.append(
                    StageReport(stage_name, requested, 0, 0, requested, False, f"exception:{type(exc).__name__}")
                )
                continue

            candidate = np.asarray(raw.image)
            if candidate.shape != before.shape or not np.isfinite(candidate.astype(np.float32)).all():
                _restore_stage_state(workspace, state)
                reports.append(StageReport(stage_name, requested, 0, 0, requested, False, "invalid_output"))
                continue

            # Commit is ROI-local.  A model can never rewrite already-protected pixels
            # or unrelated regions merely because it returned a full-frame image.
            clipped = before.copy()
            clipped[active] = candidate[active]
            workspace.primary = clipped.copy()

            if provenance_before is not None:
                current = workspace.provenance_map
                if isinstance(current, np.ndarray) and current.shape == active.shape:
                    merged = provenance_before.copy()
                    merged[active] = current[active]
                    workspace.provenance_map = merged
                else:
                    workspace.provenance_map = provenance_before.copy()

            generated = _binary(workspace.metadata.get("inpaint_generated_mask"), active.shape) > 0
            generated_pixels = int(np.count_nonzero(generated & active))
            if stage_name != "severe" and generated_pixels > 0:
                _restore_stage_state(workspace, state)
                reports.append(
                    StageReport(
                        stage_name,
                        requested,
                        0,
                        generated_pixels,
                        requested,
                        False,
                        "generation_forbidden_before_severe",
                    )
                )
                continue

            changed_mask = np.any(clipped != before, axis=2) & active
            protected_violation = changed_mask & protected_before
            if np.any(protected_violation):
                _restore_stage_state(workspace, state)
                reports.append(
                    StageReport(stage_name, requested, 0, generated_pixels, requested, False, "protected_region_modified")
                )
                continue

            observed = _binary(workspace.metadata.get("inpaint_observed_mask"), active.shape) > 0
            symmetry = _binary(workspace.metadata.get("inpaint_symmetry_mask"), active.shape) > 0
            unresolved_raw = _binary(workspace.metadata.get("inpaint_unresolved_mask"), active.shape) > 0

            # Resolution must be demonstrated by an accepted change/evidence signal;
            # an absent unresolved mask alone is not enough to claim success.
            resolution_signal = changed_mask | observed | symmetry | generated
            resolved = active & ~unresolved_raw & resolution_signal
            protected |= resolved
            remaining = active & ~resolved
            workspace.metadata["protected_region_mask"] = np.where(protected, 255, 0).astype(np.uint8)

            unresolved_pixels = int(np.count_nonzero(remaining))
            changed_pixels = int(np.count_nonzero(changed_mask))
            reason = "accepted" if np.any(resolved) else "accepted_no_progress_escalate"
            reports.append(
                StageReport(
                    stage_name,
                    requested,
                    changed_pixels,
                    generated_pixels,
                    unresolved_pixels,
                    True,
                    reason,
                )
            )
            final_details.update(dict(raw.details))

        workspace.metadata.pop("adaptive_restoration_stage_mask", None)
        workspace.metadata["adaptive_restoration_stage"] = "complete"
        workspace.metadata["protected_region_mask"] = np.where(protected, 255, 0).astype(np.uint8)
        workspace.metadata["adaptive_restoration_reports"] = [item.__dict__.copy() for item in reports]
        workspace.metadata["adaptive_restoration_remaining_mask"] = np.where(remaining, 255, 0).astype(np.uint8)
        final_details["stages"] = [item.__dict__.copy() for item in reports]
        final_details["severity_counts"] = dict(workspace.metadata.get("adaptive_severity_counts", {}))
        final_details["protected_pixels"] = int(np.count_nonzero(protected))
        final_details["remaining_pixels"] = int(np.count_nonzero(remaining))

        if not any_stage:
            final_details["reason"] = "no_local_restoration_required"
        return ExecutionResult(block.key, workspace.copy_primary(), final_details)

    cascade._adaptive_restoration_cascade = True  # type: ignore[attr-defined]
    executor._handlers[BlockKind.INPAINT] = cascade
