from __future__ import annotations

from pathlib import Path

from app.model_registry import registry_by_key
from app.model_selection import installed_plan, select_model_for_block
from app.pipeline import BlockKind


def _touch_manifest(root: Path, key: str) -> Path:
    manifest = registry_by_key()[key]
    path = root / manifest.destination
    if manifest.filename == path.name and "." in manifest.filename:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path


def test_missing_model_uses_fallback_without_error(tmp_path: Path) -> None:
    selected = select_model_for_block(BlockKind.DEBLUR, tmp_path)
    assert selected.uses_pretrained is False
    assert selected.model_key is None
    assert "OpenCV" in selected.fallback


def test_selection_respects_cpu_first_priority(tmp_path: Path) -> None:
    _touch_manifest(tmp_path, "restormer_motion_deblur")
    _touch_manifest(tmp_path, "nafnet_gopro_width32")
    selected = select_model_for_block(BlockKind.DEBLUR, tmp_path)
    assert selected.uses_pretrained is True
    assert selected.model_key == "nafnet_gopro_width32"


def test_sface_is_not_selected_without_yunet_dependency(tmp_path: Path) -> None:
    _touch_manifest(tmp_path, "opencv_sface")
    selected = select_model_for_block(BlockKind.IDENTITY_CHECK, tmp_path)
    assert selected.model_key is None


def test_sface_is_selected_when_yunet_pair_is_installed(tmp_path: Path) -> None:
    _touch_manifest(tmp_path, "opencv_sface")
    _touch_manifest(tmp_path, "opencv_yunet")
    selected = select_model_for_block(BlockKind.IDENTITY_CHECK, tmp_path)
    assert selected.model_key == "opencv_sface"


def test_yunet_is_first_cpu_landmark_choice(tmp_path: Path) -> None:
    _touch_manifest(tmp_path, "opencv_yunet")
    selected = select_model_for_block(BlockKind.LANDMARKS, tmp_path)
    assert selected.model_key == "opencv_yunet"


def test_deterministic_blocks_never_require_models(tmp_path: Path) -> None:
    assert select_model_for_block(BlockKind.IMPORT, tmp_path).model_key is None
    assert select_model_for_block(BlockKind.EXPORT, tmp_path).model_key is None


def test_installed_plan_has_one_selection_per_block(tmp_path: Path) -> None:
    selections = installed_plan(tmp_path)
    assert len(selections) == len(BlockKind)
    assert {item.block for item in selections} == set(BlockKind)
