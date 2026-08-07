from __future__ import annotations

from pathlib import Path

from app.model_selection import combined_registry, installed_plan, select_model_for_block
from app.pipeline import BlockKind


def _touch_manifest(root: Path, key: str) -> Path:
    manifest = combined_registry()[key]
    path = root / manifest.destination
    if manifest.filename == path.name and "." in manifest.filename:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Test fixtures intentionally do not match production hashes; selection
        # tests only exercise installed-path priority, so use manifests without a
        # checksum check by monkey-style replacement is unnecessary here.
        path.write_bytes(b"test")
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path


def test_missing_model_uses_fallback_without_error(tmp_path: Path) -> None:
    selected = select_model_for_block(BlockKind.DEBLUR, tmp_path)
    assert selected.uses_pretrained is False
    assert selected.model_key is None
    assert "OpenCV" in selected.fallback


def test_verified_onnx_is_first_deblur_choice(tmp_path: Path, monkeypatch) -> None:
    from app import model_selection

    _touch_manifest(tmp_path, "opencv_nafnet_deblur")
    original = model_selection.inspect_model

    def inspect_without_fixture_hash(manifest, root):
        status = original(manifest, root)
        if manifest.key == "opencv_nafnet_deblur" and status["exists"]:
            status["checksum_ok"] = True
        return status

    monkeypatch.setattr(model_selection, "inspect_model", inspect_without_fixture_hash)
    selected = select_model_for_block(BlockKind.DEBLUR, tmp_path)
    assert selected.uses_pretrained is True
    assert selected.model_key == "opencv_nafnet_deblur"


def test_sface_is_not_selected_without_yunet_dependency(tmp_path: Path) -> None:
    _touch_manifest(tmp_path, "opencv_sface")
    selected = select_model_for_block(BlockKind.IDENTITY_CHECK, tmp_path)
    assert selected.model_key is None


def test_sface_is_selected_when_yunet_pair_is_installed(tmp_path: Path, monkeypatch) -> None:
    from app import model_selection

    _touch_manifest(tmp_path, "opencv_sface")
    _touch_manifest(tmp_path, "opencv_yunet")
    original = model_selection.inspect_model

    def inspect_without_fixture_hash(manifest, root):
        status = original(manifest, root)
        if manifest.key in {"opencv_sface", "opencv_yunet"} and status["exists"]:
            status["checksum_ok"] = True
        return status

    monkeypatch.setattr(model_selection, "inspect_model", inspect_without_fixture_hash)
    selected = select_model_for_block(BlockKind.IDENTITY_CHECK, tmp_path)
    assert selected.model_key == "opencv_sface"


def test_yunet_is_first_landmark_choice(tmp_path: Path, monkeypatch) -> None:
    from app import model_selection

    _touch_manifest(tmp_path, "opencv_yunet")
    original = model_selection.inspect_model

    def inspect_without_fixture_hash(manifest, root):
        status = original(manifest, root)
        if manifest.key == "opencv_yunet" and status["exists"]:
            status["checksum_ok"] = True
        return status

    monkeypatch.setattr(model_selection, "inspect_model", inspect_without_fixture_hash)
    selected = select_model_for_block(BlockKind.LANDMARKS, tmp_path)
    assert selected.model_key == "opencv_yunet"


def test_deterministic_blocks_never_require_models(tmp_path: Path) -> None:
    assert select_model_for_block(BlockKind.IMPORT, tmp_path).model_key is None
    assert select_model_for_block(BlockKind.EXPORT, tmp_path).model_key is None


def test_installed_plan_has_one_selection_per_block(tmp_path: Path) -> None:
    selections = installed_plan(tmp_path)
    assert len(selections) == len(BlockKind)
    assert {item.block for item in selections} == set(BlockKind)
