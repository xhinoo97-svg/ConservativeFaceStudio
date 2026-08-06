from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DeblurSettings:
    denoise: int = 5
    sharpen: float = 1.0
    contrast: float = 1.0


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Immagine non valida")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def conservative_deblur(image: np.ndarray, settings: DeblurSettings) -> np.ndarray:
    """Deblur leggero senza generazione di nuovi tratti facciali."""
    source = _ensure_bgr(image)
    strength = max(1, int(settings.denoise))
    denoised = cv2.fastNlMeansDenoisingColored(
        source,
        None,
        strength,
        strength,
        7,
        21,
    )

    blurred = cv2.GaussianBlur(denoised, (0, 0), 1.2)
    amount = float(np.clip(settings.sharpen, 0.0, 2.5))
    sharpened = cv2.addWeighted(denoised, 1.0 + amount, blurred, -amount, 0)

    contrast = float(np.clip(settings.contrast, 0.6, 1.6))
    adjusted = cv2.convertScaleAbs(sharpened, alpha=contrast, beta=0)
    return adjusted


def quality_enhance(image: np.ndarray) -> np.ndarray:
    """Migliora contrasto locale e luminanza preservando il colore."""
    source = _ensure_bgr(image)
    lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(lightness)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
