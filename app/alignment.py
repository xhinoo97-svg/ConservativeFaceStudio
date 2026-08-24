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
    reprojection_error: float = 0.0


def _gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Immagine non valida")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(image, code)
    raise ValueError("Formato immagine non supportato")


def _point_array(points: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
        raise ValueError(f"{name} deve avere forma Nx2 con almeno 3 punti")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contiene valori non finiti")
    return array


def _validate_transform(matrix: np.ndarray, minimum_scale: float, maximum_scale: float) -> float:
    linear = matrix[:, :2]
    determinant = float(np.linalg.det(linear))
    scale = abs(determinant) ** 0.5
    if determinant <= 0 or not minimum_scale <= scale <= maximum_scale:
        raise ValueError(f"Scala o orientamento non plausibili: {scale:.3f}")
    return scale


def _phase_translation(moving: np.ndarray, reference: np.ndarray) -> AlignmentResult:
    """Fallback conservativo per sole traslazioni quando non esistono descrittori affidabili."""
    if moving.shape[:2] != reference.shape[:2]:
        raise ValueError("Phase correlation richiede immagini della stessa dimensione")
    moving_gray = _gray(moving).astype(np.float32)
    reference_gray = _gray(reference).astype(np.float32)
    window = cv2.createHanningWindow((moving_gray.shape[1], moving_gray.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(moving_gray, reference_gray, window)
    dx, dy = float(shift[0]), float(shift[1])
    if not np.isfinite([dx, dy, response]).all() or response < 0.02:
        raise ValueError("Dettagli insufficienti per l'allineamento")
    height, width = reference.shape[:2]
    if abs(dx) > width * 0.25 or abs(dy) > height * 0.25:
        raise ValueError("Traslazione stimata non plausibile")
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    aligned = cv2.warpAffine(
        moving,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return AlignmentResult(aligned, matrix, 0, float(np.clip(response, 0.0, 1.0)), 0.0)


def align_from_points(
    moving: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    output_shape: tuple[int, int],
    *,
    ransac_threshold: float = 3.0,
    minimum_inlier_ratio: float = 0.6,
    maximum_reprojection_error: float = 5.0,
) -> AlignmentResult:
    """Allinea mediante punti osservati con controlli RANSAC e di riproiezione."""
    if moving is None or moving.size == 0:
        raise ValueError("Immagine non valida")
    source = _point_array(source_points, "source_points")
    target = _point_array(target_points, "target_points")
    if source.shape != target.shape:
        raise ValueError("Gli insiemi di punti devono avere la stessa forma")
    height, width = output_shape
    if height <= 0 or width <= 0 or ransac_threshold <= 0:
        raise ValueError("Parametri geometrici non validi")

    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold),
        maxIters=3000,
        confidence=0.995,
        refineIters=20,
    )
    if matrix is None or inliers is None:
        raise ValueError("Trasformazione non stimabile")
    _validate_transform(matrix, 0.55, 1.8)

    inlier_mask = inliers.reshape(-1).astype(bool)
    ratio = float(np.mean(inlier_mask))
    if ratio < minimum_inlier_ratio:
        raise ValueError(f"Allineamento instabile: inlier ratio {ratio:.3f}")

    homogeneous = np.column_stack((source, np.ones(len(source), dtype=np.float32)))
    projected = homogeneous @ matrix.T
    errors = np.linalg.norm(projected - target, axis=1)
    relevant = errors[inlier_mask]
    mean_error = float(np.mean(relevant)) if relevant.size else float("inf")
    if mean_error > maximum_reprojection_error:
        raise ValueError(f"Errore di riproiezione troppo alto: {mean_error:.3f}px")

    aligned = cv2.warpAffine(
        moving,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return AlignmentResult(aligned, matrix.astype(np.float32), len(source), ratio, mean_error)


def normalize_points(points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    array = _point_array(points, "points").copy()
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("Dimensioni immagine non valide")
    array[:, 0] /= float(width)
    array[:, 1] /= float(height)
    return array


def denormalize_points(points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    array = _point_array(points, "points").copy()
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("I punti normalizzati devono essere compresi tra 0 e 1")
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("Dimensioni immagine non valide")
    array[:, 0] *= float(width)
    array[:, 1] *= float(height)
    return array


def _feature_affine(
    moving: np.ndarray,
    reference: np.ndarray,
    *,
    prefer_sift: bool,
    min_matches: int,
    minimum_inlier_ratio: float,
    maximum_reprojection_error: float,
    minimum_scale: float,
    maximum_scale: float,
) -> AlignmentResult:
    """Local-feature affine alignment that also accepts partial/different-size photos."""
    moving_gray = _gray(moving)
    reference_gray = _gray(reference)
    if prefer_sift and hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.02, edgeThreshold=12)
        norm = cv2.NORM_L2
        ratio_threshold = 0.72
    else:
        detector = cv2.ORB_create(nfeatures=3500, fastThreshold=8)
        norm = cv2.NORM_HAMMING
        ratio_threshold = 0.75
    kp_a, desc_a = detector.detectAndCompute(moving_gray, None)
    kp_b, desc_b = detector.detectAndCompute(reference_gray, None)
    if desc_a is None or desc_b is None or len(kp_a) < min_matches or len(kp_b) < min_matches:
        raise ValueError("Dettagli locali insufficienti per una reference parziale")
    pairs = cv2.BFMatcher(norm).knnMatch(desc_a, desc_b, k=2)
    good = [first for first, second in pairs if first.distance < ratio_threshold * second.distance]
    if len(good) < min_matches:
        raise ValueError(f"Reference parziale: solo {len(good)} corrispondenze affidabili")
    source = np.float32([kp_a[m.queryIdx].pt for m in good])
    target = np.float32([kp_b[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.997,
        refineIters=30,
    )
    if matrix is None or inliers is None:
        raise ValueError("Reference parziale: trasformazione locale non stimabile")
    _validate_transform(matrix, minimum_scale, maximum_scale)
    active = inliers.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(active))
    inlier_ratio = float(np.mean(active))
    if inlier_count < max(4, min_matches // 2) or inlier_ratio < minimum_inlier_ratio:
        raise ValueError(f"Reference parziale instabile: {inlier_count} inlier, ratio {inlier_ratio:.3f}")
    homogeneous = np.column_stack((source, np.ones(len(source), dtype=np.float32)))
    projected = homogeneous @ matrix.T
    errors = np.linalg.norm(projected - target, axis=1)
    reprojection_error = float(np.mean(errors[active])) if np.any(active) else float("inf")
    if reprojection_error > maximum_reprojection_error:
        raise ValueError(f"Reference parziale: errore di riproiezione {reprojection_error:.3f}px")
    height, width = reference.shape[:2]
    aligned = cv2.warpAffine(
        moving,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return AlignmentResult(aligned, matrix.astype(np.float32), len(good), inlier_ratio, reprojection_error)


def align_partial_to_reference(
    moving: np.ndarray,
    reference: np.ndarray,
    *,
    min_matches: int = 6,
) -> AlignmentResult:
    """Strict alignment for a crop showing only one facial component.

    No face detector is required. SIFT is preferred when available because partial
    crops often have too little context for ORB. The transform is accepted only with
    strong RANSAC support and low reprojection error; otherwise the caller must abstain.
    """
    return _feature_affine(
        moving,
        reference,
        prefer_sift=True,
        min_matches=max(6, int(min_matches)),
        minimum_inlier_ratio=0.50,
        maximum_reprojection_error=4.0,
        minimum_scale=0.40,
        maximum_scale=2.50,
    )


def align_to_reference(
    moving: np.ndarray,
    reference: np.ndarray,
    *,
    max_features: int = 2500,
    min_matches: int = 12,
) -> AlignmentResult:
    """Allinea con ORB+RANSAC e fallback di sola traslazione, senza sintetizzare contenuto."""
    if moving.shape[:2] != reference.shape[:2]:
        # Different-size input is common for partial facial crops. Use the stricter
        # local-feature path instead of resizing the crop and distorting geometry.
        return align_partial_to_reference(moving, reference, min_matches=max(6, min_matches // 2))
    detector = cv2.ORB_create(nfeatures=max(200, int(max_features)), fastThreshold=12)
    kp_a, desc_a = detector.detectAndCompute(_gray(moving), None)
    kp_b, desc_b = detector.detectAndCompute(_gray(reference), None)
    if desc_a is None or desc_b is None:
        return _phase_translation(moving, reference)

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    if len(good) < min_matches:
        return _phase_translation(moving, reference)

    source = np.float32([kp_a[m.queryIdx].pt for m in good])
    target = np.float32([kp_b[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=3000, confidence=0.995, refineIters=20,
    )
    if matrix is None or inliers is None:
        return _phase_translation(moving, reference)

    _validate_transform(matrix, 0.75, 1.35)
    inlier_mask = inliers.reshape(-1).astype(bool)
    inlier_ratio = float(np.mean(inlier_mask))
    if inlier_ratio < 0.35:
        return _phase_translation(moving, reference)

    homogeneous = np.column_stack((source, np.ones(len(source), dtype=np.float32)))
    projected = homogeneous @ matrix.T
    errors = np.linalg.norm(projected - target, axis=1)
    reprojection_error = float(np.mean(errors[inlier_mask]))

    height, width = reference.shape[:2]
    aligned = cv2.warpAffine(
        moving, matrix, (width, height), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return AlignmentResult(aligned, matrix, len(good), inlier_ratio, reprojection_error)


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
