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
    retention_ratio: float = 1.0
    minimum_retention: float = 0.95


def _embedding_score(
    image: np.ndarray,
    anchors: Iterable[np.ndarray],
    *,
    backend: object | None = None,
) -> tuple[float, str] | None:
    try:
        analyser = backend if backend is not None else choose_backend(prefer_embeddings=True)
        analysed = analyser.analyze(image)
        if analysed.embedding is None:
            return None
        scores: list[float] = []
        for anchor in anchors:
            other = analyser.analyze(anchor)
            if other.embedding is not None:
                scores.append(cosine_similarity(analysed.embedding, other.embedding))
        if scores:
            name = getattr(analyser, "name", None)
            if name is None:
                target = getattr(analyser, "target_name", "cpu")
                name = f"opencv-zoo-sface-{target}"
            return max(scores), str(name)
    except Exception:
        return None
    return None


def identity_anchor_score(
    image: np.ndarray,
    anchors: list[np.ndarray],
    *,
    backend: object | None = None,
) -> tuple[float, str]:
    if not anchors:
        return 1.0, "no-anchor"
    embedded = _embedding_score(image, anchors, backend=backend)
    if embedded is not None:
        return embedded
    return max(identity_similarity_proxy(image, anchor) for anchor in anchors), "lab-histogram-proxy"


def evaluate_identity_scores(
    before_score: float,
    after_score: float,
    *,
    engine: str,
    max_drop: float = 0.12,
    absolute_minimum: float = 0.25,
    minimum_retention: float = 0.95,
) -> GuardrailDecision:
    """Apply the same conservative decision rule to externally computed identity scores."""
    if not 0.0 < minimum_retention <= 1.0:
        raise ValueError("minimum_retention deve essere compreso tra 0 e 1")
    before_score = float(before_score)
    after_score = float(after_score)
    if not np.isfinite(before_score) or not np.isfinite(after_score):
        raise ValueError("Punteggi identità non validi")

    drop = float(before_score - after_score)
    if before_score > 1e-6:
        retention_ratio = float(after_score / before_score)
    else:
        retention_ratio = 1.0 if after_score >= before_score else 0.0

    retention_required = before_score >= absolute_minimum
    retained = (not retention_required) or retention_ratio >= minimum_retention
    accepted = after_score >= absolute_minimum and drop <= max_drop and retained
    reason = "accepted" if accepted else (
        f"identity regression: {after_score:.3f}, drop {drop:.3f}, retention {retention_ratio:.3f}; "
        f"limits minimum={absolute_minimum:.3f}, max_drop={max_drop:.3f}, retention>={minimum_retention:.3f}"
    )
    return GuardrailDecision(
        accepted,
        before_score,
        after_score,
        drop,
        str(engine),
        reason,
        retention_ratio,
        float(minimum_retention),
    )


def evaluate_identity_guardrail(
    before: np.ndarray,
    candidate: np.ndarray,
    anchors: list[np.ndarray],
    *,
    max_drop: float = 0.12,
    absolute_minimum: float = 0.25,
    minimum_retention: float = 0.95,
    backend: object | None = None,
) -> GuardrailDecision:
    """Rifiuta trasformazioni che riducono troppo la coerenza con fotografie osservate.

    ``backend`` permette alla pipeline di riusare il modello SFace già caricato invece
    di tentare un secondo backend o ricadere silenziosamente sul proxy LAB.
    """
    if before is None or before.size == 0 or candidate is None or candidate.size == 0:
        raise ValueError("Immagini non valide per il guardrail")
    effective = list(anchors) if anchors else [before]
    before_score, before_engine = identity_anchor_score(before, effective, backend=backend)
    after_score, after_engine = identity_anchor_score(candidate, effective, backend=backend)
    engine = after_engine if after_engine != "no-anchor" else before_engine
    return evaluate_identity_scores(
        before_score,
        after_score,
        engine=engine,
        max_drop=max_drop,
        absolute_minimum=absolute_minimum,
        minimum_retention=minimum_retention,
    )


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
