from __future__ import annotations

import math

import cv2
import numpy as np

from app.face_analysis import cosine_similarity


def _as_float_gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Immagine non valida")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(image, code)
    else:
        raise ValueError("Formato immagine non supportato")
    return gray.astype(np.float64)


def psnr(reference: np.ndarray, candidate: np.ndarray, *, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio per immagini della stessa forma."""
    if reference.shape != candidate.shape:
        raise ValueError("Le immagini devono avere la stessa forma")
    if data_range <= 0:
        raise ValueError("data_range deve essere positivo")
    left = reference.astype(np.float64)
    right = candidate.astype(np.float64)
    mse = float(np.mean((left - right) ** 2))
    if mse <= 1e-15:
        return float("inf")
    return float(10.0 * math.log10((data_range * data_range) / mse))


def structural_similarity_global(reference: np.ndarray, candidate: np.ndarray, *, data_range: float = 255.0) -> float:
    """SSIM globale leggero e deterministico; non sostituisce la SSIM windowed dei benchmark ufficiali."""
    if reference.shape != candidate.shape:
        raise ValueError("Le immagini devono avere la stessa forma")
    if data_range <= 0:
        raise ValueError("data_range deve essere positivo")
    left = _as_float_gray(reference)
    right = _as_float_gray(candidate)
    mu_x = float(left.mean())
    mu_y = float(right.mean())
    var_x = float(left.var())
    var_y = float(right.var())
    covariance = float(np.mean((left - mu_x) * (right - mu_y)))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * covariance + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if denominator <= 1e-15:
        return 1.0 if np.array_equal(reference, candidate) else 0.0
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def normalized_landmark_error(
    reference_points: np.ndarray,
    candidate_points: np.ndarray,
    *,
    normalization_distance: float | None = None,
) -> float:
    """Errore landmark medio normalizzato; con 5 punti usa la distanza inter-oculare se non specificata."""
    reference = np.asarray(reference_points, dtype=np.float64)
    candidate = np.asarray(candidate_points, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2:
        raise ValueError("Landmark non compatibili")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Landmark non finiti")
    scale = float(normalization_distance) if normalization_distance is not None else float(np.linalg.norm(reference[0] - reference[1]))
    if scale <= 1e-9:
        raise ValueError("Distanza di normalizzazione nulla")
    distances = np.linalg.norm(reference - candidate, axis=1)
    return float(np.mean(distances) / scale)


def identity_cosine_score(reference_embedding: np.ndarray, candidate_embedding: np.ndarray) -> float:
    """AFICS-style primitive: similarita' coseno tra embedding facciali normalizzati."""
    return cosine_similarity(reference_embedding, candidate_embedding)
