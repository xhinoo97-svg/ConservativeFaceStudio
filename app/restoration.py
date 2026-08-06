from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DeblurSettings:
    denoise: int = 5
    sharpen: float = 1.0
    contrast: float = 1.0
    preserve_edges: bool = True


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Immagine non valida")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Formato immagine non supportato")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def conservative_deblur(image: np.ndarray, settings: DeblurSettings) -> np.ndarray:
    """Denoise + unsharp mask limitato, senza sintesi di contenuto."""
    source = _ensure_bgr(image)
    strength = int(np.clip(settings.denoise, 0, 20))
    if strength:
        if settings.preserve_edges:
            denoised = cv2.bilateralFilter(source, 7, 20 + strength * 2, 20 + strength * 2)
        else:
            denoised = cv2.fastNlMeansDenoisingColored(source, None, strength, strength, 7, 21)
    else:
        denoised = source

    blurred = cv2.GaussianBlur(denoised, (0, 0), 1.2)
    amount = float(np.clip(settings.sharpen, 0.0, 2.0))
    sharpened = cv2.addWeighted(denoised, 1.0 + amount, blurred, -amount, 0)

    contrast = float(np.clip(settings.contrast, 0.6, 1.6))
    return cv2.convertScaleAbs(sharpened, alpha=contrast, beta=0)


def quality_enhance(image: np.ndarray, clip_limit: float = 1.7) -> np.ndarray:
    """Migliora solo la luminanza con CLAHE per limitare deviazioni cromatiche."""
    source = _ensure_bgr(image)
    lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(np.clip(clip_limit, 1.0, 3.0)), tileGridSize=(8, 8))
    enhanced_l = clahe.apply(lightness)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def detect_occlusion_candidates(image: np.ndarray) -> np.ndarray:
    """Maschera euristica conservativa di regioni molto scure/chiare o poco testurizzate.

    Non identifica semanticamente l'oggetto: produce solo candidati da confermare in UI.
    """
    source = _ensure_bgr(image)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    local_mean = cv2.GaussianBlur(gray, (0, 0), 7)
    local_sq = cv2.GaussianBlur(gray.astype(np.float32) ** 2, (0, 0), 7)
    variance = np.maximum(local_sq - local_mean.astype(np.float32) ** 2, 0)

    extreme = ((gray < 18) | (gray > 242)).astype(np.uint8) * 255
    flat = ((variance < 12) & (saturation < 18)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(extreme, flat)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def conservative_fusion(base: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fonde pixel osservati da una foto di riferimento già allineata.

    La maschera 0..255 stabilisce dove usare il riferimento; nessun pixel viene generato.
    """
    base_bgr = _ensure_bgr(base)
    ref_bgr = _ensure_bgr(reference)
    if base_bgr.shape != ref_bgr.shape:
        raise ValueError("Base e riferimento devono avere la stessa dimensione")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape != base_bgr.shape[:2]:
        raise ValueError("La maschera non coincide con l'immagine")
    alpha = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
    fused = base_bgr.astype(np.float32) * (1.0 - alpha) + ref_bgr.astype(np.float32) * alpha
    return np.clip(np.rint(fused), 0, 255).astype(np.uint8)


def identity_similarity_proxy(before: np.ndarray, after: np.ndarray) -> float:
    """Controllo deterministico non biometrico basato su istogrammi LAB.

    Serve come guardrail offline; non sostituisce un embedding facciale ArcFace/InsightFace.
    """
    left = cv2.cvtColor(_ensure_bgr(before), cv2.COLOR_BGR2LAB)
    right = cv2.cvtColor(_ensure_bgr(after), cv2.COLOR_BGR2LAB)
    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    scores = []
    for channel in range(3):
        h1 = cv2.calcHist([left], [channel], None, [64], [0, 256])
        h2 = cv2.calcHist([right], [channel], None, [64], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        scores.append((cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) + 1.0) / 2.0)
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def conservative_upscale(image: np.ndarray, scale: int = 2) -> np.ndarray:
    """Upscale deterministico senza inventare dettagli mediante interpolazione Lanczos."""
    source = _ensure_bgr(image)
    if scale not in (1, 2, 3, 4):
        raise ValueError("Il fattore di scala deve essere 1, 2, 3 o 4")
    if scale == 1:
        return source
    height, width = source.shape[:2]
    return cv2.resize(source, (width * scale, height * scale), interpolation=cv2.INTER_LANCZOS4)
