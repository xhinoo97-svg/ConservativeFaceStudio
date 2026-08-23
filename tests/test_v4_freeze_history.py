from __future__ import annotations

from scripts.verify_v4_freeze_history import FREEZE_COMMIT, FROZEN_BLOB_SHA, REQUIRED, verify


def test_v4_frozen_manifests_match_pinned_original_git_blobs() -> None:
    result = verify()
    assert set(result) == set(REQUIRED) == set(FROZEN_BLOB_SHA)
    for name, item in result.items():
        assert item["frozen_commit_sha"] == FREEZE_COMMIT
        assert item["expected_frozen_blob_sha"] == FROZEN_BLOB_SHA[name]
        assert item["current_blob_sha"] == FROZEN_BLOB_SHA[name]
        if item["history_origin_if_available"]:
            assert item["history_origin_if_available"] == FREEZE_COMMIT
