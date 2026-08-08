from __future__ import annotations

import warnings
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ReferenceRepairResult:
    image: np.ndarray
    target_mask: np.ndarray
    repaired_mask: np.ndarray
    provenance_map: np.ndarray
    requested_pixels: int
    repaired_pixels: int
    unresolved_pixels: int
    source_pixel_counts: tuple[int, ...]


@dataclass(frozen=True)
class PoseNormalizationResult:
    image: np.ndarray
    applied: bool
    roll_degrees: float
    scale: float
    supported_fraction: float
    reason: str


def _validate_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Immagine non valida")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Sono supportate immagini BGR a 3 canali")
    return image


def _validate_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=np.uint8)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape != shape:
        raise ValueError("Maschera non compatibile con l'immagine")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def face_support_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    """Limita gli interventi alla faccia osservata; senza bbox non limita l'immagine."""
    h, w = shape
    if bbox is None:
        return np.full((h, w), 255, dtype=np.uint8)
    x, y, bw, bh = (int(v) for v in bbox)
    if bw <= 0 or bh <= 0:
        return np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(round(x + bw * 0.5)), int(round(y + bh * 0.52)))
    axes = (max(2, int(round(bw * 0.50))), max(2, int(round(bh * 0.53))))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _local_sharpness(image: np.ndarray) -> np.ndarray:
    """Edge-energy locale normalizzata, usata solo per confermare blur relativo ai riferimenti."""
    gray = cv2.cvtColor(_validate_bgr(image), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    return cv2.GaussianBlur(laplacian, (0, 0), 1.5)


def _finite_mean_last_axis(values: np.ndarray) -> np.ndarray:
    """Mean over the last axis without RuntimeWarning on all-NaN pixels."""
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    count = np.sum(finite, axis=-1)
    total = np.sum(np.where(finite, array, 0.0), axis=-1, dtype=np.float32)
    result = np.full(count.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=result, where=count > 0)
    return result


def reference_consensus_occlusion_mask(
    primary: np.ndarray,
    references: list[np.ndarray],
    target_hint: np.ndarray,
    reference_masks: list[np.ndarray] | None = None,
    *,
    face_mask: np.ndarray | None = None,
    difference_threshold: float = 0.10,
    strong_difference_threshold: float = 0.20,
    agreement_threshold: float = 0.055,
    blur_deficit_threshold: float = 0.035,
    minimum_reference_sharpness: float = 0.020,
    sharpness_agreement_threshold: float = 0.020,
    maximum_fraction: float = 0.25,
) -> np.ndarray:
    """Trova coperture solo quando foto reali allineate supportano la correzione.

    Con un solo riferimento richiede anche la maschera euristica. Con almeno due
    riferimenti accetta differenze forti soltanto quando i riferimenti concordano.
    Rileva inoltre blur locale della primaria quando almeno due riferimenti concordano
    su struttura nitida nella stessa zona. Questo evita di trattare un semplice volto
    morbido come occlusione: il deficit deve essere relativo a fotografie della stessa
    persona, localmente allineate e non mascherate. Se l'area candidata e troppo grande,
    il metodo si astiene anziche modificare una porzione estesa del volto.
    """
    base = _validate_bgr(primary)
    if not references:
        return np.zeros(base.shape[:2], dtype=np.uint8)
    if any(_validate_bgr(item).shape != base.shape for item in references):
        raise ValueError("I riferimenti devono essere allineati e avere la stessa forma")

    shape = base.shape[:2]
    hint = _validate_mask(target_hint, shape) > 0
    masks = reference_masks or [np.zeros(shape, dtype=np.uint8) for _ in references]
    if len(masks) != len(references):
        raise ValueError("Numero di maschere riferimento non valido")
    masks = [_validate_mask(item, shape) for item in masks]
    support = np.ones(shape, dtype=bool) if face_mask is None else _validate_mask(face_mask, shape) > 0

    base_lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    ref_lab = np.stack(
        [cv2.cvtColor(item, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0 for item in references], axis=0
    )
    valid = np.stack(
        [
            (mask == 0) & (np.max(reference, axis=2) > 2)
            for reference, mask in zip(references, masks)
        ],
        axis=0,
    )
    values = ref_lab.copy()
    values[~valid[..., None].repeat(3, axis=3)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(values, axis=0)
        mad = np.nanmedian(np.abs(values - median[None, ...]), axis=0)
    valid_count = np.sum(valid, axis=0)
    primary_difference = _finite_mean_last_axis(np.abs(base_lab - median))
    agreement = _finite_mean_last_axis(mad)

    if len(references) == 1:
        candidate = hint & (primary_difference >= difference_threshold) & (valid_count >= 1)
    else:
        reference_agreement = (agreement <= agreement_threshold) & (valid_count >= 2)
        candidate = reference_agreement & (
            (hint & (primary_difference >= difference_threshold))
            | (primary_difference >= strong_difference_threshold)
        )

        base_sharpness = _local_sharpness(base)
        sharp_stack = np.stack([_local_sharpness(item) for item in references], axis=0)
        sharp_values = sharp_stack.copy()
        sharp_values[~valid] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_sharpness = np.nanmedian(sharp_values, axis=0)
            sharpness_mad = np.nanmedian(
                np.abs(sharp_values - median_sharpness[None, ...]), axis=0
            )
        blur_seed = (
            (valid_count >= 2)
            & (median_sharpness >= float(minimum_reference_sharpness))
            & ((median_sharpness - base_sharpness) >= float(blur_deficit_threshold))
            & (sharpness_mad <= float(sharpness_agreement_threshold))
        )
        blur_mask = blur_seed.astype(np.uint8) * 255
        if np.any(blur_seed):
            blur_mask = cv2.morphologyEx(
                blur_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
            )
            blur_mask = cv2.dilate(
                blur_mask, np.ones((5, 5), dtype=np.uint8), iterations=1
            )
            blur_mask = cv2.morphologyEx(
                blur_mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
            )
        candidate |= blur_mask > 0

    candidate &= support & np.isfinite(primary_difference)

    support_pixels = max(1, int(np.count_nonzero(support)))
    if np.count_nonzero(candidate) > int(support_pixels * maximum_fraction):
        candidate &= hint
    if np.count_nonzero(candidate) > int(support_pixels * maximum_fraction):
        return np.zeros(shape, dtype=np.uint8)

    mask = candidate.astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _absolute_quality(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(_validate_bgr(image), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    sharpness = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (0, 0), 2.0)
    exposure = 1.0 - np.clip(np.abs(gray - 0.5) / 0.5, 0.0, 1.0)
    return sharpness + 0.02 * exposure


def repair_from_observed_references(
    primary: np.ndarray,
    references: list[np.ndarray],
    target_mask: np.ndarray,
    reference_masks: list[np.ndarray] | None = None,
    *,
    feather_sigma: float = 1.2,
) -> ReferenceRepairResult:
    """Ripara soltanto con pixel di fotografie reali gia allineate; nessun inpainting generativo."""
    base = _validate_bgr(primary)
    if not references:
        raise ValueError("Serve almeno una fotografia di riferimento")
    if any(_validate_bgr(item).shape != base.shape for item in references):
        raise ValueError("I riferimenti devono avere la stessa forma della primaria")
    shape = base.shape[:2]
    target = _validate_mask(target_mask, shape) > 0
    masks = reference_masks or [np.zeros(shape, dtype=np.uint8) for _ in references]
    if len(masks) != len(references):
        raise ValueError("Numero di maschere riferimento non valido")
    masks = [_validate_mask(item, shape) for item in masks]

    scores: list[np.ndarray] = []
    valid_sources: list[np.ndarray] = []
    for reference, mask in zip(references, masks):
        valid = (mask == 0) & (np.max(reference, axis=2) > 2)
        score = _absolute_quality(reference).copy()
        score[~valid] = -np.inf
        scores.append(score)
        valid_sources.append(valid)
    score_stack = np.stack(scores, axis=0)
    best = np.argmax(score_stack, axis=0)
    best_score = np.max(score_stack, axis=0)
    repairable = target & np.isfinite(best_score)

    rows, cols = np.indices(shape)
    stacked = np.stack(references, axis=0)
    selected = stacked[best, rows, cols]
    raw_mask = repairable.astype(np.uint8) * 255
    if feather_sigma > 0 and np.any(repairable):
        alpha = cv2.GaussianBlur(raw_mask, (0, 0), float(feather_sigma)).astype(np.float32) / 255.0
        alpha *= repairable.astype(np.float32)
    else:
        alpha = repairable.astype(np.float32)
    alpha3 = alpha[..., None]
    output = np.clip(
        base.astype(np.float32) * (1.0 - alpha3) + selected.astype(np.float32) * alpha3,
        0,
        255,
    ).astype(np.uint8)
    output[~repairable] = base[~repairable]

    provenance = np.zeros(shape, dtype=np.uint16)
    provenance[repairable] = best[repairable].astype(np.uint16) + 1
    counts = tuple(int(np.count_nonzero(provenance == index + 1)) for index in range(len(references)))
    requested = int(np.count_nonzero(target))
    repaired = int(np.count_nonzero(repairable))
    return ReferenceRepairResult(
        output,
        raw_mask,
        raw_mask,
        provenance,
        requested,
        repaired,
        requested - repaired,
        counts,
    )


def conservative_roll_normalize(
    image: np.ndarray,
    landmarks5: np.ndarray,
    *,
    minimum_angle: float = 0.75,
    maximum_angle: float = 12.0,
    maximum_scale: float = 1.12,
) -> PoseNormalizationResult:
    """Corregge solo il roll usando pixel osservati; non ricostruisce lati del volto non visibili."""
    source = _validate_bgr(image)
    points = np.asarray(landmarks5, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Sono necessari 5 landmark validi")
    left_eye, right_eye = points[0], points[1]
    raw_angle = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
    if abs(raw_angle) < minimum_angle:
        return PoseNormalizationResult(source.copy(), False, raw_angle, 1.0, 1.0, "roll gia entro soglia")
    if abs(raw_angle) > maximum_angle:
        return PoseNormalizationResult(source.copy(), False, raw_angle, 1.0, 1.0, "roll troppo ampio per una correzione conservativa")

    center = tuple(((left_eye + right_eye) * 0.5).tolist())
    h, w = source.shape[:2]
    support = np.full((h, w), 255, dtype=np.uint8)
    best: tuple[np.ndarray, np.ndarray, float, float] | None = None
    for sign in (1.0, -1.0):
        for scale in np.linspace(1.0, maximum_scale, 7):
            matrix = cv2.getRotationMatrix2D(center, sign * raw_angle, float(scale))
            transformed_points = np.column_stack((points, np.ones(5, dtype=np.float32))) @ matrix.T
            residual = abs(float(transformed_points[1, 1] - transformed_points[0, 1]))
            warped_support = cv2.warpAffine(support, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            supported_fraction = float(np.mean(warped_support > 0))
            if supported_fraction >= 0.995:
                if best is None or residual < best[2]:
                    best = (matrix, warped_support, residual, float(scale))
                break
    if best is None:
        return PoseNormalizationResult(source.copy(), False, raw_angle, 1.0, 1.0, "la rotazione richiederebbe pixel non osservati")
    matrix, warped_support, _, scale = best
    result = cv2.warpAffine(source, matrix, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    supported_fraction = float(np.mean(warped_support > 0))
    return PoseNormalizationResult(result, True, raw_angle, scale, supported_fraction, "roll corretto senza sintesi")
