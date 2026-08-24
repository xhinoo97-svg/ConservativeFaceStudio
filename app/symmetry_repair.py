from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SymmetryRepairResult:
    image: np.ndarray
    repaired_mask: np.ndarray
    source_mask: np.ndarray
    confidence: float
    used: bool


def symmetry_repair(
    image: np.ndarray,
    target_mask: np.ndarray,
    reliable_mask: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    *,
    maximum_fraction: float = 0.12,
) -> SymmetryRepairResult:
    """Mirror observed opposite-side pixels as a low-confidence last resort.

    Symmetry is never treated as identity evidence. It is allowed only for a small
    lateral target with reliable mirrored source pixels. Central structures are left
    unresolved because mirroring nose/mouth centre can create unsupported anatomy.
    """
    if image is None or image.size == 0 or image.ndim != 3:
        raise ValueError("Immagine non valida")
    h, w = image.shape[:2]
    if target_mask.shape != (h, w) or reliable_mask.shape != (h, w):
        raise ValueError("Maschere non compatibili")
    x, y, fw, fh = (int(v) for v in face_bbox)
    if fw <= 0 or fh <= 0:
        raise ValueError("Bounding box non valida")

    face = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(face, (max(0, x), max(0, y)), (min(w - 1, x + fw), min(h - 1, y + fh)), 255, -1)
    target = (target_mask > 0) & (face > 0)
    face_pixels = max(1, int(np.count_nonzero(face)))
    if np.count_nonzero(target) == 0 or np.count_nonzero(target) / face_pixels > float(maximum_fraction):
        empty = np.zeros((h, w), dtype=np.uint8)
        return SymmetryRepairResult(image.copy(), empty, empty, 0.0, False)

    mid_x = float(x + fw * 0.5)
    yy, xx = np.indices((h, w))
    # Exclude central 18% of face width from symmetry synthesis.
    lateral = np.abs(xx.astype(np.float32) - mid_x) >= 0.09 * fw
    target &= lateral
    if not np.any(target):
        empty = np.zeros((h, w), dtype=np.uint8)
        return SymmetryRepairResult(image.copy(), empty, empty, 0.0, False)

    mirror_x = np.rint(2.0 * mid_x - xx).astype(np.int32)
    valid_mirror = (mirror_x >= 0) & (mirror_x < w)
    source_y = yy
    source_x = np.clip(mirror_x, 0, w - 1)
    source_reliable = reliable_mask[source_y, source_x] > 0
    usable = target & valid_mirror & source_reliable
    if int(np.count_nonzero(usable)) < 32:
        empty = np.zeros((h, w), dtype=np.uint8)
        return SymmetryRepairResult(image.copy(), empty, empty, 0.0, False)

    output = image.copy()
    output[usable] = image[source_y[usable], source_x[usable]]
    repaired = usable.astype(np.uint8) * 255
    source_mask = np.zeros((h, w), dtype=np.uint8)
    source_mask[source_y[usable], source_x[usable]] = 255
    coverage = float(np.count_nonzero(usable) / max(1, np.count_nonzero(target)))
    confidence = float(np.clip(0.45 * coverage, 0.0, 0.45))
    return SymmetryRepairResult(output, repaired, source_mask, confidence, True)
