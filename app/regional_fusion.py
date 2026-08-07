from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.alignment import quality_map


@dataclass(frozen=True)
class RegionDecision:
    name: str
    source_index: int
    primary_score: float
    selected_score: float
    improvement: float
    area_pixels: int


def _ellipse_mask(shape: tuple[int, int], center: tuple[float, float], axes: tuple[float, float]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = int(round(center[0])), int(round(center[1]))
    ax = max(2, int(round(axes[0])))
    ay = max(2, int(round(axes[1])))
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return mask


def facial_region_masks(shape: tuple[int, int], landmarks5: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    points = np.asarray(landmarks5, dtype=np.float32)
    if points.shape != (5, 2):
        raise ValueError("Sono necessari 5 landmark facciali")
    x, y, w, h = bbox
    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_dist = max(8.0, float(np.linalg.norm(right_eye - left_eye)))
    mouth_center = (left_mouth + right_mouth) / 2.0
    masks = {
        "left_eye": _ellipse_mask(shape, tuple(left_eye), (0.22 * eye_dist, 0.14 * h)),
        "right_eye": _ellipse_mask(shape, tuple(right_eye), (0.22 * eye_dist, 0.14 * h)),
        "nose": _ellipse_mask(shape, tuple(nose), (0.18 * w, 0.20 * h)),
        "mouth": _ellipse_mask(shape, tuple(mouth_center), (0.28 * w, 0.14 * h)),
        "face": _ellipse_mask(shape, (x + 0.5 * w, y + 0.52 * h), (0.46 * w, 0.50 * h)),
    }
    # Evita che la regione generica del volto sovrascriva le componenti più specifiche.
    specific = cv2.bitwise_or(masks["left_eye"], masks["right_eye"])
    specific = cv2.bitwise_or(specific, masks["nose"])
    specific = cv2.bitwise_or(specific, masks["mouth"])
    masks["face"] = cv2.bitwise_and(masks["face"], cv2.bitwise_not(specific))
    return masks


def _mean_score(score: np.ndarray, mask: np.ndarray) -> float:
    selected = score[mask > 0]
    return float(np.mean(selected)) if selected.size else 0.0


def regional_reference_fusion(
    images: list[np.ndarray],
    occlusion_masks: list[np.ndarray],
    landmarks5: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    minimum_improvement: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, tuple[RegionDecision, ...]]:
    """Sostituisce regioni solo quando una foto osservata è misurabilmente migliore della primaria."""
    if len(images) < 2:
        raise ValueError("Servono almeno una primaria e un riferimento")
    if len(images) != len(occlusion_masks):
        raise ValueError("Numero immagini/maschere non compatibile")
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError("Le immagini devono essere allineate e avere la stessa forma")
    regions = facial_region_masks(shape[:2], landmarks5, bbox)
    scores = [quality_map(image, mask) for image, mask in zip(images, occlusion_masks)]
    output = images[0].copy()
    provenance = np.zeros(shape[:2], dtype=np.uint16)
    decisions: list[RegionDecision] = []

    for name, region_mask in regions.items():
        primary_score = _mean_score(scores[0], region_mask)
        candidates = [_mean_score(score, region_mask) for score in scores]
        best_index = int(np.argmax(candidates))
        best_score = float(candidates[best_index])
        improvement = best_score - primary_score
        area = int(np.count_nonzero(region_mask))
        if best_index > 0 and improvement >= minimum_improvement and area > 0:
            # Maschera sfumata solo per evitare cuciture; i pixel provengono comunque da una foto osservata.
            feather = cv2.GaussianBlur(region_mask, (0, 0), 2.0).astype(np.float32) / 255.0
            alpha = feather[..., None]
            output = np.clip(output.astype(np.float32) * (1.0 - alpha) + images[best_index].astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            provenance[region_mask > 0] = best_index
        decisions.append(RegionDecision(name, best_index if improvement >= minimum_improvement else 0, primary_score, best_score, improvement, area))
    return output, provenance, tuple(decisions)
