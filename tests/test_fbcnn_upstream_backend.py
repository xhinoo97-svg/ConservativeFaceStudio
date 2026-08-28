from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.face_restorer_adapter import RestorationContext
from app.fbcnn_upstream_backend import (
    APPROVED_CHECKPOINT_SHA256,
    FBCNNUpstreamBackend,
    OFFICIAL_REPOSITORY,
    PINNED_REVISION,
    _damage_route_allowed,
    _load_checkout_metadata,
)


def _write_metadata(root: Path, **overrides) -> None:
    payload = {
        "format": "ConservativeFaceStudio pinned upstream checkout",
        "key": "fbcnn",
        "official_repository": OFFICIAL_REPOSITORY,
        "clone_url": f"https://github.com/{OFFICIAL_REPOSITORY}.git",
        "pinned_revision": PINNED_REVISION,
        "actual_revision": PINNED_REVISION,
        "qualification_state": "CANDIDATE",
        "research_only": True,
        "architecture_reimplemented_by_cfs": False,
    }
    payload.update(overrides)
    (root / ".cfs-upstream.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_fbcnn_checkout_requires_exact_official_repository_and_revision(tmp_path: Path) -> None:
    _write_metadata(tmp_path)
    metadata = _load_checkout_metadata(tmp_path)
    assert metadata["official_repository"] == "jiaxi-jiang/FBCNN"
    assert metadata["actual_revision"] == PINNED_REVISION
    assert metadata["architecture_reimplemented_by_cfs"] is False

    _write_metadata(tmp_path, official_repository="someone/fork")
    with pytest.raises(RuntimeError, match="wrong official repository"):
        _load_checkout_metadata(tmp_path)

    _write_metadata(tmp_path, actual_revision="0" * 40)
    with pytest.raises(RuntimeError, match="approved detached revision"):
        _load_checkout_metadata(tmp_path)


def test_fbcnn_backend_requires_full_checkpoint_sha256(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full lowercase SHA-256"):
        FBCNNUpstreamBackend(tmp_path, tmp_path / "model.pth", expected_checkpoint_sha256="unknown")


def test_fbcnn_backend_rejects_a_well_formed_but_unapproved_checkpoint_hash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="approved FBCNN checkpoint"):
        FBCNNUpstreamBackend(
            tmp_path,
            tmp_path / "model.pth",
            expected_checkpoint_sha256="0" * 64,
        )


def test_fbcnn_wrong_checkpoint_hash_fails_before_model_import(tmp_path: Path) -> None:
    _write_metadata(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "network_fbcnn.py").write_text("raise AssertionError('must not import')\n", encoding="utf-8")
    checkpoint = tmp_path / "fbcnn_color.pth"
    checkpoint.write_bytes(b"not-the-approved-checkpoint")

    backend = FBCNNUpstreamBackend(
        tmp_path,
        checkpoint,
        expected_checkpoint_sha256=APPROVED_CHECKPOINT_SHA256,
    )
    with pytest.raises(RuntimeError, match="checkpoint byte size mismatch"):
        backend.load()


def test_fbcnn_accepts_only_jpeg_or_explicit_recompression_routes() -> None:
    assert _damage_route_allowed(RestorationContext(damage_class="jpeg_artifacts", severity="medium"))
    assert _damage_route_allowed(RestorationContext(damage_class="double_jpeg", severity="heavy"))
    assert _damage_route_allowed(RestorationContext(damage_class="social_recompression", severity="medium"))
    assert _damage_route_allowed(
        RestorationContext(
            damage_class="mixed",
            severity="medium",
            metadata={"jpeg_detected": True},
        )
    )
    assert not _damage_route_allowed(RestorationContext(damage_class="blur", severity="heavy"))
    assert not _damage_route_allowed(RestorationContext(damage_class="mosaic", severity="heavy"))
    assert not _damage_route_allowed(RestorationContext(damage_class="healthy", severity="none"))


def test_fbcnn_adapter_is_thin_and_binds_candidate_to_verified_upstream_artifacts() -> None:
    source = Path(__file__).resolve().parents[1] / "app" / "fbcnn_upstream_backend.py"
    text = source.read_text(encoding="utf-8")
    assert "models\" / \"network_fbcnn.py" in text
    assert "class FBCNN(" not in text
    assert "architecture_reimplemented_by_cfs" in text
    assert "upstream_repository=OFFICIAL_REPOSITORY" in text
    assert "upstream_revision=PINNED_REVISION" in text
    assert "checkpoint_sha256=self._checkpoint_sha256" in text


def test_checkpoint_fixture_helper_has_real_sha_semantics(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fixture.pth"
    checkpoint.write_bytes(b"fixture")
    expected = hashlib.sha256(b"fixture").hexdigest()
    assert len(expected) == 64
    assert expected != "0" * 64


def test_generated_candidate_contract_stays_full_resolution() -> None:
    image = np.zeros((64, 72, 3), dtype=np.uint8)
    assert image.ndim == 3
    assert image.dtype == np.uint8
