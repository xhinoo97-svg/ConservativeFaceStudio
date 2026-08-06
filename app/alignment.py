from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    image: np.ndarray
    matrix: np.ndarray
    matches: int
    inlier_ratio: float


def _gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Immagine non valida")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(image, code)
    raise ValueError("Formato immagine non supportato")


def align_to_reference(
    moving: np.ndarray,
    reference: np.ndarray,
    *,
    max_features: int = 2500,
    min_matches: int = 12,
) -> AlignmentResult:
    """Allinea con ORB+RANSAC senza sintetizzare contenuto fuori dall'immagine sorgente."""
    if moving.shape[:2] != reference.shape[:2]:
        raise ValueError("Le immagini devono avere la stessa dimensione prima dell'allineamento")
    detector = cv2.ORB_create(nfeatures=max(200, int(max_features)), fastThreshold=12)
    kp_a, desc_a = detector.detectAndCompute(_gray(moving), None)
    kp_b, desc_b = detector.detectAndCompute(_gray(reference), None)
    if desc_a is None or desc_b is None:
        raise ValueError("Dettagli insufficienti per l'allineamento")

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    if len(good) < min_matches:
        raise ValueError(f"Corrispondenze insufficienti: {len(good)}")

    source = np.float32([kp_a[m.queryIdx].pt for m in good])
    target = np.float32([kp_b[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=3000, confidence=0.995, refineIters=20,
    )
    if matrix is None or inliers is None:
        raise ValueError("Trasformazione non stimabile")

    linear = matrix[:, :2]
    determinant = float(np.linalg.det(linear))
    scale = abs(determinant) ** 0.5
    if not 0.75 <= scale <= 1.35:
        raise ValueError(f"Scala di allineamento non plausibile: {scale:.3f}")
    inlier_ratio = float(inliers.mean())
    if inlier_ratio < 0.35:
        raise ValueError(f"Allineamento instabile: inlier ratio {inlier_ratio:.3f}")

    height, width = reference.shape[:2]
    aligned = cv2.warpAffine(
        moving, matrix, (width, height), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return AlignmentResult(aligned, matrix, len(good), inlier_ratio)


def quality_map(image: np.ndarray, occlusion_mask: np.ndarray | None = None) -> np.ndarray:
    """Punteggio locale deterministico: nitidezza, esposizione e assenza di occlusione."""
    gray = _gray(image).astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = cv2.GaussianBlur(np.abs(lap), (0, 0), 3.0)
    sharpness /= float(sharpness.max() + 1e-6)
    exposure = 1.0 - np.clip(np.abs(gray - 0.5) / 0.5, 0.0, 1.0)
    score = 0.7 * sharpness + 0.3 * exposure
    if occlusion_mask is not None:
        if occlusion_mask.shape != gray.shape:
            raise ValueError("Maschera e immagine non compatibili")
        score *= 1.0 - (occlusion_mask.astype(np.float32) / 255.0)
    return np.clip(score, 0.0, 1.0)


def select_best_observed_pixels(
    images: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Seleziona per pixel la sorgente osservata con qualità maggiore; nessuna generazione."""
    if not images:
        raise ValueError("Serve almeno un'immagine")
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError("Tutte le immagini devono avere la stessa forma")
    if masks is None:
        masks = [np.zeros(shape[:2], dtype=np.uint8) for _ in images]
    if len(masks) != len(images):
        raise ValueError("Numero di maschere non valido")

    scores = np.stack([quality_map(image, mask) for image, mask in zip(images, masks)], axis=0)
    source_index = np.argmax(scores, axis=0).astype(np.uint16)
    stacked = np.stack(images, axis=0)
    rows, cols = np.indices(shape[:2])
    result = stacked[source_index, rows, cols]
    return result, source_index
