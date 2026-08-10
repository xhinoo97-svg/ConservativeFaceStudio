from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.execution import ExecutionResult
from app.pipeline import BlockKind
from app.severity_aware_deblur_policy import classify_blur

CLEAN = np.uint8(0)
LIGHT = np.uint8(1)
MEDIUM = np.uint8(2)
SEVERE = np.uint8(3)
MISSING = np.uint8(4)
NON_RECOVERABLE_FROM_THIS_IMAGE = np.uint8(5)


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
            cv2.ellipse(mask, ((x1 + x2) // 2, (y1 + y2) // 2), (max(1, (x2-x1)//2), max(1, (y2-y1)//2)), 0, 0, 360, 255, -1)
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


def build_severity_map(workspace) -> np.ndarray:
    """Build a local severity map from frozen observed evidence, never from aesthetics.

    Occlusion component size controls obstruction severity. Blur is added only when the
    existing blur classifier detects an image-level blur condition; low-detail skin by
    itself is therefore not enough to create a restoration target.
    """
    image = workspace.primary
    shape = image.shape[:2]
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
        x = int(stats[label, cv2.CC_STAT_LEFT]); y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH]); bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_frame = x <= 0 or y <= 0 or x + bw >= shape[1] or y + bh >= shape[0]
        level = MISSING if touches_frame and ratio >= 0.05 else (LIGHT if ratio <= 0.012 else MEDIUM if ratio <= 0.08 else SEVERE)
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
        roi = cv2.morphologyEx(blur_roi.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
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
    return severity


def _mask_for_stage(severity: np.ndarray, name: str) -> np.ndarray:
    if name == "light":
        active = severity == LIGHT
    elif name == "medium":
        active = severity == MEDIUM
    elif name == "severe":
        active = (severity == SEVERE) | (severity == MISSING) | (severity == NON_RECOVERABLE_FROM_THIS_IMAGE)
    else:
        raise ValueError(f"Unknown restoration stage: {name}")
    return np.where(active, 255, 0).astype(np.uint8)


def install_adaptive_restoration_cascade(executor) -> None:
    """Wrap the final INPAINT handler after every reference/same-canvas policy is wired."""
    original = executor._handlers.get(BlockKind.INPAINT)
    if original is None or getattr(original, "_adaptive_restoration_cascade", False):
        return

    def cascade(block, parameters: dict[str, Any]) -> ExecutionResult:
        workspace = executor.workspace
        severity = build_severity_map(workspace)
        reports: list[StageReport] = []
        final_details: dict[str, Any] = {"engine": "adaptive-light-medium-severe", "adaptive_cascade": True}
        any_stage = False

        for stage_name in ("light", "medium", "severe"):
            stage_mask = _mask_for_stage(severity, stage_name)
            requested = int(np.count_nonzero(stage_mask))
            if requested == 0:
                reports.append(StageReport(stage_name, 0, 0, 0, 0, True, "not_required"))
                continue

            any_stage = True
            before = workspace.copy_primary()
            provenance_before = None if workspace.provenance_map is None else workspace.provenance_map.copy()
            workspace.metadata["adaptive_restoration_stage"] = stage_name
            workspace.metadata["adaptive_restoration_stage_mask"] = stage_mask.copy()

            p = dict(parameters)
            p["allow_verified_generative"] = stage_name == "severe"
            if stage_name != "severe":
                p["maximum_generated_face_fraction"] = 0.0
                p["maximum_generated_target_fraction"] = 0.0

            raw = original(block, p)
            candidate = raw.image.copy()
            if candidate.shape != before.shape or not np.isfinite(candidate.astype(np.float32)).all():
                workspace.primary = before.copy()
                if provenance_before is not None:
                    workspace.provenance_map = provenance_before
                reports.append(StageReport(stage_name, requested, 0, 0, requested, False, "invalid_output"))
                continue

            active = stage_mask > 0
            clipped = before.copy()
            clipped[active] = candidate[active]
            workspace.primary = clipped.copy()

            if provenance_before is not None and isinstance(workspace.provenance_map, np.ndarray) and workspace.provenance_map.shape == active.shape:
                merged_prov = provenance_before.copy()
                merged_prov[active] = workspace.provenance_map[active]
                workspace.provenance_map = merged_prov

            generated = _binary(workspace.metadata.get("inpaint_generated_mask"), active.shape)
            generated_pixels = int(np.count_nonzero((generated > 0) & active))
            if stage_name != "severe" and generated_pixels > 0:
                workspace.primary = before.copy()
                if provenance_before is not None:
                    workspace.provenance_map = provenance_before
                reports.append(StageReport(stage_name, requested, 0, generated_pixels, requested, False, "generation_forbidden_before_severe"))
                continue

            unresolved = _binary(workspace.metadata.get("inpaint_unresolved_mask"), active.shape)
            unresolved_pixels = int(np.count_nonzero((unresolved > 0) & active))
            changed = int(np.count_nonzero(np.any(clipped != before, axis=2) & active))
            reports.append(StageReport(stage_name, requested, changed, generated_pixels, unresolved_pixels, True, "accepted"))
            final_details.update(dict(raw.details))

        workspace.metadata.pop("adaptive_restoration_stage_mask", None)
        workspace.metadata["adaptive_restoration_stage"] = "complete"
        workspace.metadata["adaptive_restoration_reports"] = [item.__dict__.copy() for item in reports]
        final_details["stages"] = [item.__dict__.copy() for item in reports]
        final_details["severity_counts"] = dict(workspace.metadata.get("adaptive_severity_counts", {}))

        if not any_stage:
            final_details["reason"] = "no_local_restoration_required"
        return ExecutionResult(block.key, workspace.copy_primary(), final_details)

    cascade._adaptive_restoration_cascade = True  # type: ignore[attr-defined]
    executor._handlers[BlockKind.INPAINT] = cascade
