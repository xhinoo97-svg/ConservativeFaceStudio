from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from app.face_analysis import choose_backend, cosine_similarity
from app.restoration import identity_similarity_proxy


@dataclass(frozen=True)
class GuardrailDecision:
    accepted: bool
    score_before: float
    score_after: float
    score_drop: float
    engine: str
    reason: str


def _embedding_score(image: np.ndarray, anchors: Iterable[np.ndarray]) -> tuple[float, str] | None:
    try:
        backend = choose_backend(prefer_embeddings=True)
        analysed = backend.analyze(image)
        if analysed.embedding is None:
            return None
        scores: list[float] = []
        for anchor in anchors:
            other = backend.analyze(anchor)
            if other.embedding is not None:
                scores.append(cosine_similarity(analysed.embedding, other.embedding))
        if scores:
            return max(scores), backend.name
    except Exception:
        return None
    return None


def identity_anchor_score(image: np.ndarray, anchors: list[np.ndarray]) -> tuple[float, str]:
    if not anchors:
        return 1.0, "no-anchor"
    embedded = _embedding_score(image, anchors)
    if embedded is not None:
        return embedded
    return max(identity_similarity_proxy(image, anchor) for anchor in anchors), "lab-histogram-proxy"


def evaluate_identity_guardrail(
    before: np.ndarray,
    candidate: np.ndarray,
    anchors: list[np.ndarray],
    *,
    max_drop: float = 0.12,
    absolute_minimum: float = 0.25,
) -> GuardrailDecision:
    """Rifiuta un blocco solo quando peggiora in modo significativo la coerenza con immagini osservate.

    Gli anchor devono essere fotografie reali della stessa persona. Se non esistono riferimenti,
    l'immagine precedente viene usata come anchor conservativo per rilevare cambiamenti estremi.
    """
    if before is None or before.size == 0 or candidate is None or candidate.size == 0:
        raise ValueError("Immagini non valide per il guardrail")
    effective = list(anchors) if anchors else [before]
    before_score, before_engine = identity_anchor_score(before, effective)
    after_score, after_engine = identity_anchor_score(candidate, effective)
    engine = after_engine if after_engine != "no-anchor" else before_engine
    drop = float(before_score - after_score)
    accepted = after_score >= absolute_minimum and drop <= max_drop
    reason = "accepted" if accepted else (
        f"identity regression: {after_score:.3f}, drop {drop:.3f}, "
        f"limits minimum={absolute_minimum:.3f}, max_drop={max_drop:.3f}"
    )
    return GuardrailDecision(accepted, float(before_score), float(after_score), drop, engine, reason)


@dataclass(frozen=True)
class ValidationMetrics:
    psnr: float
    sharpness: float
    identity_score: float


def validation_metrics(candidate: np.ndarray, ground_truth: np.ndarray) -> ValidationMetrics:
    if candidate.shape[:2] != ground_truth.shape[:2]:
        candidate = cv2.resize(candidate, (ground_truth.shape[1], ground_truth.shape[0]), interpolation=cv2.INTER_AREA)
    psnr = float(cv2.PSNR(candidate, ground_truth))
    gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY) if candidate.ndim == 3 else candidate
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    identity = float(identity_similarity_proxy(candidate, ground_truth))
    return ValidationMetrics(psnr, sharpness, identity)


def synthetic_degradations(image: np.ndarray) -> dict[str, np.ndarray]:
    """Piccolo validation set deterministico creato dall'immagine stessa; nessun dataset redistribuito."""
    if image is None or image.size == 0:
        raise ValueError("Immagine non valida")
    blurred = cv2.GaussianBlur(image, (7, 7), 1.8)
    rng = np.random.default_rng(12345)
    noise = rng.normal(0.0, 8.0, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 45])
    jpeg = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else image.copy()
    occluded = image.copy()
    h, w = image.shape[:2]
    cv2.rectangle(occluded, (w * 2 // 5, h * 2 // 5), (w * 3 // 5, h * 3 // 5), (0, 0, 0), -1)
    return {"blur": blurred, "noise": noisy, "jpeg": jpeg, "occlusion": occluded}
