from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


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
    specific = cv2.bitwise_or(masks["left_eye"], masks["right_eye"])
    specific = cv2.bitwise_or(specific, masks["nose"])
    specific = cv2.bitwise_or(specific, masks["mouth"])
    masks["face"] = cv2.bitwise_and(masks["face"], cv2.bitwise_not(specific))
    return masks


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32) / 255.0
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def _regional_quality(image: np.ndarray, occlusion_mask: np.ndarray, region_mask: np.ndarray) -> float:
    """Punteggio regionale confrontabile tra immagini, senza normalizzare via il massimo di ciascuna sorgente."""
    active = region_mask > 0
    if not np.any(active):
        return 0.0
    if occlusion_mask.shape != region_mask.shape:
        raise ValueError("Maschera di occlusione non compatibile con la regione")

    gray = _gray(image)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    sharpness = float(np.mean(laplacian[active]))
    exposure = 1.0 - np.clip(np.abs(gray[active] - 0.5) / 0.5, 0.0, 1.0)
    exposure_score = float(np.mean(exposure)) if exposure.size else 0.0
    visible_fraction = 1.0 - float(np.mean(occlusion_mask[active].astype(np.float32) / 255.0))

    # La nitidezza assoluta deve restare confrontabile tra sorgenti: normalizzarla per-image
    # rende indistinguibili una foto sfocata e una nitida se entrambe hanno il proprio massimo locale.
    score = (sharpness + 0.03 * exposure_score) * max(0.0, visible_fraction)
    return float(score)


def regional_reference_fusion(
    images: list[np.ndarray],
    occlusion_masks: list[np.ndarray],
    landmarks5: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    minimum_improvement: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, tuple[RegionDecision, ...]]:
    """Ripara solo pixel coperti della primaria usando regioni osservate migliori.

    La selezione della sorgente resta regionale, ma in strict mode un riferimento non
    può riscrivere pixel già osservati e validi della foto primaria. Il trasferimento è
    quindi limitato all'intersezione tra regione semantica, occlusione della primaria e
    pixel non occlusi del riferimento scelto. Il feathering resta interno a tale area,
    così immagine e provenance coincidono esattamente sui pixel modificati.
    """
    if len(images) < 2:
        raise ValueError("Servono almeno una primaria e un riferimento")
    if len(images) != len(occlusion_masks):
        raise ValueError("Numero immagini/maschere non compatibile")
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError("Le immagini devono essere allineate e avere la stessa forma")
    if any(mask.shape != shape[:2] for mask in occlusion_masks):
        raise ValueError("Le maschere di occlusione devono avere la stessa forma delle immagini")

    regions = facial_region_masks(shape[:2], landmarks5, bbox)
    output = images[0].copy()
    provenance = np.zeros(shape[:2], dtype=np.uint16)
    decisions: list[RegionDecision] = []
    primary_occluded = occlusion_masks[0] > 0

    for name, region_mask in regions.items():
        candidates = [
            _regional_quality(image, occlusion_mask, region_mask)
            for image, occlusion_mask in zip(images, occlusion_masks)
        ]
        primary_score = float(candidates[0])
        best_index = int(np.argmax(candidates))
        best_score = float(candidates[best_index])
        improvement = best_score - primary_score
        area = int(np.count_nonzero(region_mask))
        selected_index = 0

        if best_index > 0 and improvement >= minimum_improvement and area > 0:
            source_visible = occlusion_masks[best_index] == 0
            transfer_mask = (region_mask > 0) & primary_occluded & source_visible
            if np.any(transfer_mask):
                binary_transfer = transfer_mask.astype(np.uint8) * 255
                feather = cv2.GaussianBlur(binary_transfer, (0, 0), 2.0).astype(np.float32) / 255.0
                feather[~transfer_mask] = 0.0
                alpha = feather[..., None]
                output = np.clip(
                    output.astype(np.float32) * (1.0 - alpha)
                    + images[best_index].astype(np.float32) * alpha,
                    0,
                    255,
                ).astype(np.uint8)
                provenance[transfer_mask] = best_index
                selected_index = best_index

        decisions.append(RegionDecision(name, selected_index, primary_score, best_score, improvement, area))
    return output, provenance, tuple(decisions)
