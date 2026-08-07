from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransitionAssessment:
    safe: bool
    reason: str
    mean_absolute_change: float
    added_clipped_fraction: float
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
    max_added_clipped_fraction: float = 0.08,
) -> TransitionAssessment:
    """Guardrail deterministico per strict mode.

    Non valuta la qualita' estetica. Rifiuta solo output chiaramente corrotti:
    dimensioni inattese, dtype incompatibili, metriche non finite, forte shift
    globale di luminanza o nuovo clipping introdotto dall'operazione.
    """
    if before is None or after is None or before.size == 0 or after.size == 0:
        return TransitionAssessment(False, "immagine vuota", 0.0, 0.0, 0.0)
    if before.dtype != np.uint8 or after.dtype != np.uint8:
        return TransitionAssessment(False, "dtype non uint8", 0.0, 0.0, 0.0)
    if before.ndim != after.ndim:
        return TransitionAssessment(False, "numero dimensioni cambiato", 0.0, 0.0, 0.0)
    if not allow_resize and before.shape != after.shape:
        return TransitionAssessment(False, "dimensioni cambiate in un blocco non-resize", 0.0, 0.0, 0.0)
    if allow_resize and before.ndim == 3 and after.ndim == 3 and before.shape[2] != after.shape[2]:
        return TransitionAssessment(False, "numero canali cambiato", 0.0, 0.0, 0.0)

    before_clipped = float(np.mean((before <= 0) | (before >= 255)))
    after_clipped = float(np.mean((after <= 0) | (after >= 255)))
    added_clipped = max(0.0, after_clipped - before_clipped)

    before_luma = _luma(before)
    after_luma = _luma(after)
    if before_luma.shape != after_luma.shape:
        mean_change = float(abs(float(after_luma.mean()) - float(before_luma.mean())))
    else:
        mean_change = float(np.mean(np.abs(after_luma - before_luma)))
    luma_shift = float(abs(float(after_luma.mean()) - float(before_luma.mean())))

    if not np.isfinite([added_clipped, mean_change, luma_shift]).all():
        return TransitionAssessment(False, "metriche non finite", mean_change, added_clipped, luma_shift)
    if luma_shift > float(max_luma_shift):
        return TransitionAssessment(False, f"shift luminanza eccessivo: {luma_shift:.2f}", mean_change, added_clipped, luma_shift)
    if added_clipped > float(max_added_clipped_fraction):
        return TransitionAssessment(False, f"nuovo clipping eccessivo: {added_clipped:.3f}", mean_change, added_clipped, luma_shift)
    return TransitionAssessment(True, "ok", mean_change, added_clipped, luma_shift)
