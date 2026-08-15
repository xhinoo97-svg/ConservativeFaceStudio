from __future__ import annotations

import numpy as np

import app.validation as validation


def _image() -> np.ndarray:
    return np.full((32, 32, 3), 120, dtype=np.uint8)


def test_guardrail_rejects_candidate_below_relative_retention(monkeypatch) -> None:
    scores = iter([(0.80, "test"), (0.74, "test")])
    monkeypatch.setattr(validation, "identity_anchor_score", lambda *args, **kwargs: next(scores))
    decision = validation.evaluate_identity_guardrail(
        _image(),
        _image(),
        [_image()],
        max_drop=0.10,
        absolute_minimum=0.20,
        minimum_retention=0.95,
    )
    assert decision.accepted is False
    assert decision.retention_ratio < 0.95


def test_guardrail_accepts_candidate_above_relative_retention(monkeypatch) -> None:
    scores = iter([(0.80, "test"), (0.77, "test")])
    monkeypatch.setattr(validation, "identity_anchor_score", lambda *args, **kwargs: next(scores))
    decision = validation.evaluate_identity_guardrail(
        _image(),
        _image(),
        [_image()],
        max_drop=0.05,
        absolute_minimum=0.20,
        minimum_retention=0.95,
    )
    assert decision.accepted is True
    assert decision.retention_ratio >= 0.95
