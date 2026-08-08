from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DeblurSettings:
    denoise: int = 5
    sharpen: float = 0.2
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


def quality_enhance(image: np.ndarray, clip_limit: float = 1.7, blend: float = 0.2) -> np.ndarray:
    """Migliora la luminanza con CLAHE attenuato per evitare sovra-contrasto e falsi dettagli."""
    source = _ensure_bgr(image)
    lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(np.clip(clip_limit, 1.0, 3.0)), tileGridSize=(8, 8))
    enhanced_l = clahe.apply(lightness)
    mix = float(np.clip(blend, 0.0, 1.0))
    if mix < 1.0:
        enhanced_l = cv2.addWeighted(lightness, 1.0 - mix, enhanced_l, mix, 0)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def detail_reliability_map(
    image: np.ndarray,
    occlusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate where the *observed input* contains recoverable local detail.

    This map is computed before learned deblurring.  It is deliberately separate from
    the occlusion mask: a blurred face is not an occlusion, but it must not become a
    high-confidence donor merely because a restoration network later sharpens it.
    The score uses only inexpensive local statistics (standard deviation, Sobel and
    Laplacian energy) so it is safe on the target 8th-generation Intel laptop.
    """
    source = _ensure_bgr(image)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mean = cv2.GaussianBlur(gray, (0, 0), 3.0)
    mean_sq = cv2.GaussianBlur(gray * gray, (0, 0), 3.0)
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 2.0)
    lap = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)), (0, 0), 2.0)

    # Fixed scaling is intentional.  Per-image normalization would make a completely
    # blurred face look "reliable" relative to itself.  Values below ~3 are treated
    # as essentially detail-free; 20+ corresponds to strong observed structure.
    energy = 0.55 * local_std + 0.30 * (gradient / 4.0) + 0.15 * (lap / 4.0)
    reliability = np.clip((energy - 3.0) / 17.0, 0.0, 1.0)

    if occlusion_mask is not None:
        mask = np.asarray(occlusion_mask)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape != gray.shape:
            raise ValueError("Occlusion mask non compatibile con l'immagine")
        reliability[mask > 0] = 0.0

    return np.rint(reliability * 255.0).astype(np.uint8)


def detect_occlusion_candidates(image: np.ndarray) -> np.ndarray:
    """Conservative multi-signal candidate mask for stickers/scribbles/obscuration.

    This is deliberately not the final occlusion decision. It produces a broad,
    deterministic proposal that must later be constrained by face parsing and
    confirmed against aligned same-identity references. Besides black/white and
    flat regions, it detects locally implausible chroma and thin high-contrast
    marks, which are common for stickers and drawn scribbles.
    """
    source = _ensure_bgr(image)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    saturation = hsv[:, :, 1]

    local_mean = cv2.GaussianBlur(gray, (0, 0), 7)
    local_sq = cv2.GaussianBlur(gray.astype(np.float32) ** 2, (0, 0), 7)
    variance = np.maximum(local_sq - local_mean.astype(np.float32) ** 2, 0)

    extreme = ((gray < 18) | (gray > 242)).astype(np.uint8) * 255
    flat = ((variance < 10) & (saturation < 24)).astype(np.uint8) * 255

    lab_f = lab.astype(np.float32)
    local_lab = cv2.GaussianBlur(lab_f, (0, 0), 5.0)
    chroma_delta = np.linalg.norm(lab_f[:, :, 1:3] - local_lab[:, :, 1:3], axis=2)
    chroma_outlier = ((chroma_delta > 22.0) & (saturation > 48)).astype(np.uint8) * 255

    local_luma_delta = cv2.absdiff(gray, local_mean)
    edges = cv2.Canny(gray, 70, 150)
    edge_band = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    scribble = ((local_luma_delta > 38) & (edge_band > 0)).astype(np.uint8) * 255

    mask = cv2.bitwise_or(extreme, flat)
    mask = cv2.bitwise_or(mask, chroma_outlier)
    mask = cv2.bitwise_or(mask, scribble)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)


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
