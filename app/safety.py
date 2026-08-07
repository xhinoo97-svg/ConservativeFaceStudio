from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransitionAssessment:
    safe: bool
    reason: str
    mean_absolute_change: float
    clipped_fraction: float
    mean_luma_shift: float


def _luma(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Formato immagine non supportato")
    bgr = image[..., :3].astype(np.float32)
    return 0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]


def assess_transition(
    before: np.ndarray,
    after: np.ndarray,
    *,
    allow_resize: bool = False,
    max_luma_shift: float = 35.0,
    max_clipped_fraction: float = 0.10,
) -> TransitionAssessment:
    """Guardrail deterministico per strict mode.

    Non decide se un restauro e' bello: rifiuta solo output chiaramente corrotti,
    dimensioni inattese, dtype incompatibili, valori non finiti o forti shift globali.
    """
    if before is None or after is None or before.size == 0 or after.size == 0:
        return TransitionAssessment(False, "immagine vuota", 0.0, 0.0, 0.0)
    if after.dtype != np.uint8:
        return TransitionAssessment(False, "dtype output non uint8", 0.0, 0.0, 0.0)
    if before.ndim != after.ndim:
        return TransitionAssessment(False, "numero dimensioni cambiato", 0.0, 0.0, 0.0)
    if not allow_resize and before.shape != after.shape:
        return TransitionAssessment(False, "dimensioni cambiate in un blocco non-resize", 0.0, 0.0, 0.0)

    if allow_resize and before.ndim == 3 and after.ndim == 3 and before.shape[2] != after.shape[2]:
        return TransitionAssessment(False, "numero canali cambiato", 0.0, 0.0, 0.0)

    clipped = float(np.mean((after <= 0) | (after >= 255)))
    before_luma = _luma(before)
    after_luma = _luma(after)
    if before_luma.shape != after_luma.shape:
        # Per upscale confronta statistiche globali, non pixel-per-pixel.
        mean_change = float(abs(float(after_luma.mean()) - float(before_luma.mean())))
    else:
        mean_change = float(np.mean(np.abs(after_luma - before_luma)))
    luma_shift = float(abs(float(after_luma.mean()) - float(before_luma.mean())))

    if not np.isfinite([clipped, mean_change, luma_shift]).all():
        return TransitionAssessment(False, "metriche non finite", mean_change, clipped, luma_shift)
    if luma_shift > float(max_luma_shift):
        return TransitionAssessment(False, f"shift luminanza eccessivo: {luma_shift:.2f}", mean_change, clipped, luma_shift)
    if clipped > float(max_clipped_fraction):
        return TransitionAssessment(False, f"clipping eccessivo: {clipped:.3f}", mean_change, clipped, luma_shift)
    return TransitionAssessment(True, "ok", mean_change, clipped, luma_shift)
