from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import app.primary_anchor_policy as primary_anchor
from app.execution import BlockExecutionError
from app.identity_anchor_v4_hardening import (
    _face_local_identity_bridge_sources,
    _require_real_sface_result,
)


def test_shared_background_cannot_become_identity_bridge_when_face_region_differs(monkeypatch) -> None:
    monkeypatch.setattr(
        primary_anchor,
        "detect_occlusion_candidates",
        lambda image: np.zeros(image.shape[:2], dtype=np.uint8),
    )
    main = np.full((96, 96, 3), 120, dtype=np.uint8)
    other = main.copy()
    bbox = (40, 40, 16, 16)
    # Similar grayscale luminance, clearly different chroma. A whole-canvas test can
    # be dominated by the identical background while the face-local LAB test cannot.
    main[40:56, 40:56] = (200, 100, 50)
    other[40:56, 40:56] = (20, 120, 70)

    assert primary_anchor._same_canvas_match(main, other) is True
    assert primary_anchor._face_local_same_canvas_identity_match(main, other, bbox) is False


def test_face_local_identity_bridge_accepts_observed_same_canvas_face(monkeypatch) -> None:
    monkeypatch.setattr(
        primary_anchor,
        "detect_occlusion_candidates",
        lambda image: np.zeros(image.shape[:2], dtype=np.uint8),
    )
    main = np.full((96, 96, 3), 120, dtype=np.uint8)
    main[36:60, 38:58] = (90, 125, 155)
    same = main.copy()

    assert primary_anchor._face_local_same_canvas_identity_match(main, same, (36, 36, 24, 24)) is True


def test_v4_uses_only_explicit_face_local_identity_bridge_sources() -> None:
    workspace = SimpleNamespace(
        metadata={
            "same_canvas_primary_anchor": {
                "restored_source_index": 0,
                "matched_original_reference_indices": [1, 2],
                "identity_bridge_original_reference_indices": [2],
                "identity_bridge_requires_face_local_observed_agreement": True,
            }
        }
    )
    assert _face_local_identity_bridge_sources(workspace) == {2}

    workspace.metadata["same_canvas_primary_anchor"].pop(
        "identity_bridge_requires_face_local_observed_agreement"
    )
    assert _face_local_identity_bridge_sources(workspace) == set()


def test_v4_final_identity_rejects_proxy_even_with_nonempty_scores() -> None:
    proxy = SimpleNamespace(details={"engine": "lab-histogram-proxy", "scores": [0.91]})
    with pytest.raises(BlockExecutionError, match="SFace reale"):
        _require_real_sface_result(proxy)

    sface = SimpleNamespace(details={"engine": "opencv-zoo-sface-cpu", "scores": [0.40]})
    _require_real_sface_result(sface)


def test_v4_final_identity_rejects_missing_or_empty_sface_evidence() -> None:
    with pytest.raises(BlockExecutionError, match="evidenza strutturata"):
        _require_real_sface_result(SimpleNamespace(details=None))

    with pytest.raises(BlockExecutionError, match="senza confronti SFace"):
        _require_real_sface_result(SimpleNamespace(details={"engine": "opencv-zoo-sface-cpu", "scores": []}))
