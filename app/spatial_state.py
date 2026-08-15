from __future__ import annotations

from typing import Any

import cv2
import numpy as np


_SPATIAL_TOKENS = ("mask", "map", "support", "confidence", "authority", "eligibility", "occlusion", "reliability")
_CONTINUOUS_TOKENS = ("confidence", "reliability", "severity", "quality")


def _is_spatial_key(key: str) -> bool:
    return any(token in key.lower() for token in _SPATIAL_TOKENS)


def _resize_array(value: np.ndarray, source_shape: tuple[int, int], target_shape: tuple[int, int], key: str) -> np.ndarray:
    item = np.asarray(value)
    if item.ndim < 2 or item.shape[:2] != source_shape:
        return value
    interpolation = cv2.INTER_LINEAR if any(token in key.lower() for token in _CONTINUOUS_TOKENS) else cv2.INTER_NEAREST
    return cv2.resize(item, (target_shape[1], target_shape[0]), interpolation=interpolation).astype(item.dtype, copy=False)


def _resize_value(value: Any, source_shape: tuple[int, int], target_shape: tuple[int, int], key: str) -> Any:
    if isinstance(value, np.ndarray):
        return _resize_array(value, source_shape, target_shape, key)
    if isinstance(value, list):
        return [_resize_value(item, source_shape, target_shape, key) for item in value]
    if isinstance(value, tuple):
        return tuple(_resize_value(item, source_shape, target_shape, key) for item in value)
    if isinstance(value, dict):
        return {child: _resize_value(item, source_shape, target_shape, f"{key}.{child}") for child, item in value.items()}
    return value


def resize_workspace_spatial_state(workspace, source_shape: tuple[int, int], target_shape: tuple[int, int]) -> list[str]:
    """Keep all spatial evidence/authority state registered to transformed MAIN."""
    if source_shape == target_shape:
        return []
    changed: list[str] = []
    if isinstance(workspace.provenance_map, np.ndarray) and workspace.provenance_map.shape[:2] == source_shape:
        workspace.provenance_map = cv2.resize(workspace.provenance_map, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint16)
        changed.append("provenance_map")
    resized_occlusion = []
    for index, mask in enumerate(workspace.occlusion_masks):
        updated = _resize_array(mask, source_shape, target_shape, "occlusion_mask") if isinstance(mask, np.ndarray) else mask
        resized_occlusion.append(updated)
        if updated is not mask:
            changed.append(f"occlusion_masks[{index}]")
    workspace.occlusion_masks = resized_occlusion
    for key, value in list(workspace.metadata.items()):
        if not _is_spatial_key(key):
            continue
        updated = _resize_value(value, source_shape, target_shape, key)
        workspace.metadata[key] = updated
        if updated is not value:
            changed.append(key)
    scale_x = target_shape[1] / source_shape[1]
    scale_y = target_shape[0] / source_shape[0]
    bbox = workspace.metadata.get("primary_bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x, y, width, height = (float(item) for item in bbox)
        workspace.metadata["primary_bbox"] = (int(round(x * scale_x)), int(round(y * scale_y)), int(round(width * scale_x)), int(round(height * scale_y)))
        changed.append("primary_bbox")
    landmarks = workspace.metadata.get("primary_landmarks5")
    if isinstance(landmarks, np.ndarray) and landmarks.ndim == 2 and landmarks.shape[1] == 2:
        points = landmarks.astype(np.float32, copy=True)
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y
        workspace.metadata["primary_landmarks5"] = points
        changed.append("primary_landmarks5")
    return changed
