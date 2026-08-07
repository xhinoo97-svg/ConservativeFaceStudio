from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MildFrontalizationResult:
    image: np.ndarray
    applied: bool
    matrix: np.ndarray
    changed_mask: np.ndarray
    transformed_landmarks: np.ndarray
    yaw_degrees: float
    strength: float
    max_landmark_displacement: float
    supported_fraction: float
    reason: str


def _validate_inputs(
    image: np.ndarray,
    landmarks5: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Immagine BGR non valida")
    points = np.asarray(landmarks5, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Sono necessari 5 landmark facciali validi")
    x, y, w, h = (int(v) for v in bbox)
    if w <= 0 or h <= 0:
        raise ValueError("Bounding box facciale non valida")
    return image, points, (x, y, w, h)


def _least_squares_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack((source, np.ones(len(source), dtype=np.float32)))
    coeff, _, _, _ = np.linalg.lstsq(design.astype(np.float64), target.astype(np.float64), rcond=None)
    return coeff.T.astype(np.float32)


def _face_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    height, width = shape
    x, y, w, h = bbox
    center = (int(round(x + 0.5 * w)), int(round(y + 0.52 * h)))
    axes = (max(2, int(round(0.48 * w))), max(2, int(round(0.52 * h))))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def conservative_mild_frontal_affine(
    image: np.ndarray,
    landmarks5: np.ndarray,
    bbox: tuple[int, int, int, int],
    yaw_degrees: float,
    *,
    minimum_abs_yaw: float = 2.0,
    maximum_abs_yaw: float = 12.0,
    maximum_strength: float = 0.45,
    maximum_landmark_displacement_fraction: float = 0.065,
    minimum_supported_fraction: float = 0.985,
) -> MildFrontalizationResult:
    """Apply a small deterministic 2-D symmetry correction using only observed pixels.

    This is deliberately not 3-D face synthesis.  It is enabled only for mild yaw,
    moves the five observed anchors toward a symmetric configuration, estimates one
    global affine transform by least squares, and composites only transformed pixels
    that are actually supported by the source image.  The caller still applies the
    normal identity guardrail after the block.
    """
    source_image, points, box = _validate_inputs(image, landmarks5, bbox)
    yaw = float(yaw_degrees)
    identity = np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    empty = np.zeros(source_image.shape[:2], dtype=np.uint8)

    abs_yaw = abs(yaw)
    if abs_yaw < float(minimum_abs_yaw):
        return MildFrontalizationResult(
            source_image.copy(), False, identity, empty, points.copy(), yaw, 0.0, 0.0, 1.0,
            "yaw già entro la zona frontale",
        )
    if abs_yaw > float(maximum_abs_yaw):
        return MildFrontalizationResult(
            source_image.copy(), False, identity, empty, points.copy(), yaw, 0.0, 0.0, 1.0,
            "yaw troppo ampio per una correzione 2-D conservativa",
        )

    eye_mid_x = float((points[0, 0] + points[1, 0]) * 0.5)
    mouth_half = max(1.0, float(abs(points[4, 0] - points[3, 0]) * 0.5))
    target = points.copy()
    target[0, 1] = target[1, 1] = float((points[0, 1] + points[1, 1]) * 0.5)
    target[2, 0] = eye_mid_x
    target[3, 0] = eye_mid_x - mouth_half
    target[4, 0] = eye_mid_x + mouth_half
    target[3, 1] = target[4, 1] = float((points[3, 1] + points[4, 1]) * 0.5)

    strength = min(1.0, abs_yaw / float(maximum_abs_yaw)) * float(maximum_strength)
    target = points * (1.0 - strength) + target * strength
    matrix = _least_squares_affine(points, target)
    transformed = np.column_stack((points, np.ones(5, dtype=np.float32))) @ matrix.T
    displacement = np.linalg.norm(transformed - points, axis=1)
    max_displacement = float(np.max(displacement))
    max_allowed = float(max(box[2], box[3])) * float(maximum_landmark_displacement_fraction)
    if max_displacement > max_allowed:
        return MildFrontalizationResult(
            source_image.copy(), False, identity, empty, points.copy(), yaw, strength,
            max_displacement, 1.0, "spostamento landmark oltre il limite conservativo",
        )

    height, width = source_image.shape[:2]
    warped = cv2.warpAffine(
        source_image,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    support = cv2.warpAffine(
        np.full((height, width), 255, dtype=np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    face = _face_mask((height, width), box)
    usable = cv2.bitwise_and(face, support)
    face_pixels = max(1, int(np.count_nonzero(face)))
    supported_fraction = float(np.count_nonzero(usable) / face_pixels)
    if supported_fraction < float(minimum_supported_fraction):
        return MildFrontalizationResult(
            source_image.copy(), False, identity, empty, points.copy(), yaw, strength,
            max_displacement, supported_fraction, "la trasformazione richiederebbe pixel non osservati",
        )

    # Feather only the edge of the already-observed transformed face region.  Pixels
    # outside this mask remain byte-for-byte identical to the input.
    alpha = cv2.GaussianBlur(usable, (0, 0), 1.2).astype(np.float32) / 255.0
    alpha *= (usable > 0).astype(np.float32)
    alpha3 = alpha[..., None]
    result = np.clip(
        source_image.astype(np.float32) * (1.0 - alpha3) + warped.astype(np.float32) * alpha3,
        0,
        255,
    ).astype(np.uint8)
    changed = np.where(alpha > 0.001, 255, 0).astype(np.uint8)

    return MildFrontalizationResult(
        result,
        True,
        matrix,
        changed,
        transformed.astype(np.float32),
        yaw,
        strength,
        max_displacement,
        supported_fraction,
        "mild yaw corretto con warp 2-D di soli pixel osservati",
    )


def warp_auxiliary_map(
    auxiliary: np.ndarray,
    matrix: np.ndarray,
    changed_mask: np.ndarray,
    *,
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """Warp provenance/confidence only where the frontal transform changed pixels."""
    if auxiliary.ndim != 2 or changed_mask.shape != auxiliary.shape:
        raise ValueError("Mappa ausiliaria non compatibile")
    height, width = auxiliary.shape
    warped = cv2.warpAffine(
        auxiliary,
        np.asarray(matrix, dtype=np.float32),
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    result = auxiliary.copy()
    active = changed_mask > 0
    result[active] = warped[active]
    return result
