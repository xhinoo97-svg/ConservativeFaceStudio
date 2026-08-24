from __future__ import annotations

from pathlib import Path

from app.project import OperationRecord, ProjectDocument, load_project, save_project, sha256_file


def test_project_roundtrip(tmp_path: Path) -> None:
    project = ProjectDocument(
        name="restauro",
        sources=["foto1.png", "foto2.png"],
        accepted_blocks=["import", "deblur"],
        operations=[OperationRecord(block="deblur", parameters={"denoise": 5})],
    )
    path = tmp_path / "project.cfs.json"
    save_project(project, path)
    restored = load_project(path)
    assert restored.name == project.name
    assert restored.sources == project.sources
    assert restored.operations[0].conservative is True
    assert restored.operations[0].parameters["denoise"] == 5


def test_project_roundtrip_keeps_recovery_metadata(tmp_path: Path) -> None:
    project = ProjectDocument(
        name="recovery",
        sources=["main.jpg", "ref.jpg"],
        metadata={"status": "running", "last_checkpoint": "checkpoints/05_align.png"},
    )
    path = tmp_path / "recovery.cfs.json"
    save_project(project, path)

    restored = load_project(path)

    assert restored.metadata["status"] == "running"
    assert restored.metadata["last_checkpoint"] == "checkpoints/05_align.png"


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"conservative-face-studio")
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64


def test_rejects_unknown_project_version(tmp_path: Path) -> None:
    path = tmp_path / "future.json"
    path.write_text('{"name":"x","version":999}', encoding="utf-8")
    try:
        load_project(path)
    except ValueError as exc:
        assert "999" in str(exc)
    else:
        raise AssertionError("Expected unsupported version error")
