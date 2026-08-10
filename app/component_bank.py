from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ComponentCoverage:
    source_index: int
    component: str
    coverage: float
    usable: bool


def warped_support_mask(
    source_shape: tuple[int, int],
    matrix: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Project only actually observed source pixels into primary-image coordinates."""
    sh, sw = (int(v) for v in source_shape)
    th, tw = (int(v) for v in target_shape)
    if sh <= 0 or sw <= 0 or th <= 0 or tw <= 0:
        raise ValueError("Dimensioni immagine non valide")
    transform = np.asarray(matrix, dtype=np.float32)
    if transform.shape != (2, 3) or not np.isfinite(transform).all():
        raise ValueError("Matrice affine non valida")
    source = np.full((sh, sw), 255, dtype=np.uint8)
    return cv2.warpAffine(
        source,
        transform,
        (tw, th),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def canonical_component_masks(
    image_shape: tuple[int, int],
    landmarks5: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> dict[str, np.ndarray]:
    """Build tight component ROIs for full and partial reference photographs.

    Every mask is an evidence routing region, not a shape prior.  It only determines
    where an already-observed reference crop is allowed to contribute.  Eyes and brows
    are kept separate so an eye-only or brow-only crop cannot silently claim the other
    structure. Lower-face regions are likewise separated into philtrum, mouth, chin and
    jaw for partial-reference restoration.
    """
    height, width = (int(v) for v in image_shape)
    points = np.asarray(landmarks5, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Sono necessari 5 landmark validi")
    x, y, w, h = (int(v) for v in bbox)
    if w <= 0 or h <= 0:
        raise ValueError("Bounding box facciale non valida")

    def blank() -> np.ndarray:
        return np.zeros((height, width), dtype=np.uint8)

    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_distance = max(4.0, float(np.linalg.norm(right_eye - left_eye)))
    mouth_distance = max(4.0, float(np.linalg.norm(right_mouth - left_mouth)))

    masks: dict[str, np.ndarray] = {}

    eye_axes = (max(3, int(round(0.28 * eye_distance))), max(3, int(round(0.15 * h))))
    for name, center in (("left_eye", left_eye), ("right_eye", right_eye)):
        mask = blank()
        cv2.ellipse(mask, tuple(np.round(center).astype(int)), eye_axes, 0, 0, 360, 255, -1)
        masks[name] = mask

    # Five-point landmarks do not locate eyebrow hairs directly.  These are therefore
    # conservative routing bands positioned relative to the corresponding observed eye
    # and face scale; they never synthesize or geometrically reshape the brow itself.
    brow_half_width = max(4, int(round(0.30 * eye_distance)))
    brow_half_height = max(2, int(round(0.055 * h)))
    brow_offset = max(4, int(round(0.13 * h)))
    for name, center in (("left_brow", left_eye), ("right_brow", right_eye)):
        cx = int(round(center[0]))
        cy = int(round(center[1] - brow_offset))
        mask = blank()
        cv2.ellipse(
            mask,
            (cx, cy),
            (brow_half_width, brow_half_height),
            0,
            0,
            360,
            255,
            -1,
        )
        # Keep brow routing disjoint from the eye routing region.
        eye_name = "left_eye" if name == "left_brow" else "right_eye"
        mask[masks[eye_name] > 0] = 0
        masks[name] = mask

    nose_mask = blank()
    nose_center = (int(round(nose[0])), int(round(nose[1] + 0.02 * h)))
    cv2.ellipse(
        nose_mask,
        nose_center,
        (max(4, int(round(0.20 * eye_distance))), max(5, int(round(0.145 * h)))),
        0,
        0,
        360,
        255,
        -1,
    )
    masks["nose"] = nose_mask

    mouth_center_array = (left_mouth + right_mouth) * 0.5
    mouth_center = tuple(np.round(mouth_center_array).astype(int))
    mouth_mask = blank()
    cv2.ellipse(
        mouth_mask,
        mouth_center,
        (max(5, int(round(0.60 * mouth_distance))), max(4, int(round(0.105 * h)))),
        0,
        0,
        360,
        255,
        -1,
    )
    masks["mouth"] = mouth_mask

    philtrum = blank()
    philtrum_top = int(round(nose[1] + 0.10 * h))
    philtrum_bottom = int(round(mouth_center_array[1] - 0.07 * h))
    philtrum_half_width = max(3, int(round(0.16 * mouth_distance)))
    if philtrum_bottom > philtrum_top:
        cv2.rectangle(
            philtrum,
            (max(0, int(round(nose[0])) - philtrum_half_width), max(0, philtrum_top)),
            (min(width - 1, int(round(nose[0])) + philtrum_half_width), min(height - 1, philtrum_bottom)),
            255,
            -1,
        )
    masks["philtrum"] = philtrum

    chin = blank()
    chin_top = int(round(mouth_center_array[1] + 0.10 * h))
    chin_bottom = min(height - 1, y + int(round(0.93 * h)))
    chin_half_width = max(6, int(round(0.42 * w)))
    center_x = int(round(mouth_center_array[0]))
    if chin_bottom > chin_top:
        cv2.ellipse(
            chin,
            (center_x, int(round((chin_top + chin_bottom) * 0.5))),
            (chin_half_width, max(4, int(round((chin_bottom - chin_top) * 0.52)))),
            0,
            0,
            360,
            255,
            -1,
        )
    masks["chin"] = chin

    eye_mid_y = float((left_eye[1] + right_eye[1]) * 0.5)
    mouth_mid_y = float(mouth_center_array[1])
    face_mid_x = float((left_eye[0] + right_eye[0]) * 0.5)

    for name, x1, x2 in (
        ("left_cheek", x + 0.05 * w, face_mid_x - 0.05 * w),
        ("right_cheek", face_mid_x + 0.05 * w, x + 0.95 * w),
    ):
        mask = blank()
        p1 = (int(round(max(0, x1))), int(round(max(0, eye_mid_y + 0.06 * h))))
        p2 = (int(round(min(width - 1, x2))), int(round(min(height - 1, mouth_mid_y + 0.05 * h))))
        if p2[0] > p1[0] and p2[1] > p1[1]:
            cv2.rectangle(mask, p1, p2, 255, -1)
        masks[name] = mask

    forehead = blank()
    cv2.rectangle(
        forehead,
        (max(0, x + int(round(0.15 * w))), max(0, y + int(round(0.03 * h)))),
        (min(width - 1, x + int(round(0.85 * w))), min(height - 1, int(round(eye_mid_y - 0.08 * h)))),
        255,
        -1,
    )
    # Forehead is broad skin evidence; do not double-count the dedicated brows.
    forehead[masks["left_brow"] > 0] = 0
    forehead[masks["right_brow"] > 0] = 0
    masks["forehead"] = forehead

    jaw = blank()
    cv2.rectangle(
        jaw,
        (max(0, x + int(round(0.10 * w))), min(height - 1, int(round(mouth_mid_y + 0.07 * h)))),
        (min(width - 1, x + int(round(0.90 * w))), min(height - 1, y + int(round(0.98 * h)))),
        255,
        -1,
    )
    jaw[chin > 0] = 0
    masks["jaw"] = jaw
    return masks


def component_coverage(
    support_mask: np.ndarray,
    component_masks: dict[str, np.ndarray],
    *,
    source_index: int,
    minimum_coverage: float = 0.18,
) -> tuple[ComponentCoverage, ...]:
    support = np.asarray(support_mask)
    if support.ndim != 2:
        raise ValueError("Support mask non valida")
    result: list[ComponentCoverage] = []
    for name, region in component_masks.items():
        if region.shape != support.shape:
            raise ValueError("Component mask e support mask non compatibili")
        area = int(np.count_nonzero(region))
        if area <= 0:
            coverage = 0.0
        else:
            coverage = float(np.count_nonzero((region > 0) & (support > 0)) / area)
        result.append(ComponentCoverage(int(source_index), str(name), coverage, coverage >= float(minimum_coverage)))
    return tuple(result)


def build_component_bank(
    support_masks: list[np.ndarray],
    landmarks5: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    source_indices: list[int] | None = None,
    minimum_coverage: float = 0.18,
) -> dict[str, list[ComponentCoverage]]:
    if not support_masks:
        return {}
    shape = support_masks[0].shape
    if any(mask.shape != shape for mask in support_masks):
        raise ValueError("Le support mask devono avere la stessa forma")
    components = canonical_component_masks(shape, landmarks5, bbox)
    if source_indices is None:
        source_indices = list(range(len(support_masks)))
    if len(source_indices) != len(support_masks):
        raise ValueError("Numero source_indices non compatibile")

    bank: dict[str, list[ComponentCoverage]] = {name: [] for name in components}
    for mask, source_index in zip(support_masks, source_indices):
        for item in component_coverage(mask, components, source_index=source_index, minimum_coverage=minimum_coverage):
            if item.usable:
                bank[item.component].append(item)
    for values in bank.values():
        values.sort(key=lambda item: item.coverage, reverse=True)
    return bank
