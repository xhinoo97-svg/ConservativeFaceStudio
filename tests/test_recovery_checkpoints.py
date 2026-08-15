from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.block_artifacts import BlockArtifactArchive


def test_block_checkpoint_is_persisted_atomically_for_crash_recovery(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoints"
    archive = BlockArtifactArchive(directory)
    image = np.full((24, 32, 3), 90, dtype=np.uint8)

    first = archive.record("import", "Import", image, {"immutable": True})
    second = archive.record("align", "Alignment", image + 1, {"references": 2})

    manifest = json.loads((directory / "checkpoint-manifest.json").read_text(encoding="utf-8"))
    assert manifest["latest"] == second.filename
    assert [item["block"] for item in manifest["snapshots"]] == ["import", "align"]
    assert (directory / first.filename).is_file()
    assert (directory / second.filename).is_file()


def test_replaced_checkpoint_updates_persistent_manifest(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoints"
    archive = BlockArtifactArchive(directory)
    archive.record("fusion", "Fusion", np.zeros((12, 12, 3), dtype=np.uint8))
    replacement = archive.replace_last(
        np.full((12, 12, 3), 255, dtype=np.uint8),
        {"rolled_back": True},
    )

    manifest = json.loads((directory / "checkpoint-manifest.json").read_text(encoding="utf-8"))
    assert manifest["snapshots"][-1]["sha256"] == replacement.sha256
    assert manifest["snapshots"][-1]["details"]["rolled_back"] is True
