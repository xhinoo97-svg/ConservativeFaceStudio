from __future__ import annotations

from scripts.verify_v4_freeze_history import REQUIRED, verify


def test_v4_frozen_manifests_are_original_git_blobs() -> None:
    result = verify()
    assert set(result) == set(REQUIRED)
    for item in result.values():
        assert len(item["introduced_commit_sha"]) == 40
        assert item["introduced_blob_sha"] == item["current_blob_sha"]
