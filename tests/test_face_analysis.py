from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.face_analysis import InsightFaceBackend, cosine_similarity


def test_insightface_backend_never_downloads_missing_pack(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="non trovato"):
        InsightFaceBackend(root=tmp_path)
    assert list(tmp_path.rglob("*")) == []


def test_cosine_similarity_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(3, dtype=np.float32), np.ones(4, dtype=np.float32))


def test_cosine_similarity_is_bounded() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(1.0)
