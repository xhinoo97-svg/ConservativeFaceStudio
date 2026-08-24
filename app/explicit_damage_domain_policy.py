from __future__ import annotations

from functools import wraps

import numpy as np


_INSTALLED = False


def install_explicit_damage_domain_policy() -> None:
    """Make explicit observed damage authoritative over the approximate face ellipse.

    ``repair_observed_target`` already requires a trusted aligned donor, an explicit
    damage target and an observed support mask.  Clipping that target again with the
    coarse elliptical ``face_support_mask`` can incorrectly drop jaw, hairline and face
    edge pixels.  This policy removes only that geometric veto while preserving the
    original replacement budget.

    The implementation temporarily removes the bbox only for the duration of the core
    call so the existing, well-tested repair/fusion implementation remains the single
    source of truth.  The requested fraction is remapped to preserve the same absolute
    safety cap it would have had on the original face ellipse (or the explicit target if
    that is larger). No new target pixels are created.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from app import observed_target_repair_runtime as runtime

    original = runtime.repair_observed_target

    @wraps(original)
    def repair_observed_target_explicit_domain(
        workspace,
        image: np.ndarray,
        *,
        minimum_reliability: int = 0,
        agreement_colour_threshold: float = 24.0,
        maximum_face_fraction: float = 1.0,
    ):
        shape = workspace.primary.shape[:2]
        target = runtime._target_mask(workspace, shape) > 0
        if not np.any(target):
            return original(
                workspace,
                image,
                minimum_reliability=minimum_reliability,
                agreement_colour_threshold=agreement_colour_threshold,
                maximum_face_fraction=maximum_face_fraction,
            )

        bbox_raw = workspace.metadata.get("primary_bbox")
        bbox = tuple(int(v) for v in bbox_raw) if bbox_raw is not None else None
        original_face = runtime.face_support_mask(shape, bbox) > 0
        original_face_pixels = max(1, int(np.count_nonzero(original_face)))
        target_pixels = int(np.count_nonzero(target))
        target_outside_face = int(np.count_nonzero(target & ~original_face))

        fraction = float(np.clip(maximum_face_fraction, 0.0, 1.0))
        safety_domain_pixels = max(original_face_pixels, target_pixels)
        desired_cap = min(target_pixels, int(round(safety_domain_pixels * fraction)))
        image_pixels = max(1, int(shape[0] * shape[1]))
        remapped_fraction = float(np.clip(desired_cap / image_pixels, 0.0, 1.0))

        had_bbox = "primary_bbox" in workspace.metadata
        old_bbox = workspace.metadata.get("primary_bbox")
        workspace.metadata["primary_bbox"] = None
        try:
            result, provenance, details = original(
                workspace,
                image,
                minimum_reliability=minimum_reliability,
                agreement_colour_threshold=agreement_colour_threshold,
                maximum_face_fraction=remapped_fraction,
            )
        finally:
            if had_bbox:
                workspace.metadata["primary_bbox"] = old_bbox
            else:
                workspace.metadata.pop("primary_bbox", None)

        details = dict(details)
        details.update(
            {
                "explicit_target_overrides_face_template": True,
                "target_pixels_outside_face_template": target_outside_face,
                "original_face_template_pixels": original_face_pixels,
                "explicit_target_pixels": target_pixels,
                "preserved_absolute_repair_cap_pixels": desired_cap,
                "requested_maximum_face_fraction": fraction,
            }
        )
        return result, provenance, details

    runtime.repair_observed_target = repair_observed_target_explicit_domain
    _INSTALLED = True
