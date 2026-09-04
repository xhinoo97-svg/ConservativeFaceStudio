from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "config" / "fbcnn-development-evidence.json"
READINESS = ROOT / "config" / "paper-quality-readiness.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_fbcnn_development_evidence_is_bound_to_existing_artifact() -> None:
    payload = _read(EVIDENCE)
    assert payload["schema_version"] == 1
    assert payload["evidence_tier"] == "DEVELOPMENT"
    assert payload["identity_count"] == 1
    assert payload["workflow"] == {
        "run_id": 32674085939,
        "head_branch": "research/paper-quality-local-v2",
        "head_sha": "7dfeb0a855f3f7c0840693bb2c03c25bc498d4eb",
        "conclusion": "SUCCESS",
    }
    artifact = payload["artifact"]
    assert artifact["id"] == 9502200502
    assert artifact["size_bytes"] == 11500474
    assert artifact["archive_sha256"] == (
        "365251ee8b17dc31099569d328e52439fd6440e869f0ddbe16c4cb4116112842"
    )
    assert artifact["file_count"] == 38


def test_fbcnn_checkpoint_and_source_match_the_pinned_adapter() -> None:
    payload = _read(EVIDENCE)
    assert payload["model"]["official_repository"] == "jiaxi-jiang/FBCNN"
    assert payload["model"]["official_source_commit"] == (
        "54d1831927506b3247e2d4d245abb4f4dab1a1cd"
    )
    assert payload["model"]["architecture_reimplemented_by_cfs"] is False
    assert payload["checkpoint"] == {
        "asset": "fbcnn_color.pth",
        "official_release": "v1.0",
        "size_bytes": 287755111,
        "sha256": "8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8",
        "separate_terms_state": "NOT_VERIFIED",
    }


def test_all_six_development_profiles_improve_quality_and_pass_identity() -> None:
    payload = _read(EVIDENCE)
    cases = payload["cases"]
    assert len(cases) == 6
    assert len({case["profile_id"] for case in cases}) == 6
    for case in cases:
        assert case["decision"] == "PASS"
        assert case["psnr_fbcnn"] > case["psnr_degraded"]
        assert case["ssim_fbcnn"] > case["ssim_degraded"]
        assert case["sface_clean_vs_fbcnn"] >= 0.363
        assert len(case["final_sha256"]) == 64

    summary = payload["summary"]
    assert summary["completed_cases"] == 6
    assert summary["restoration_pass_count"] == 6
    assert summary["error_cases"] == 0
    assert summary["wrong_person_final_pixels"] == 0
    assert summary["provenance_violations"] == 0


def test_development_evidence_cannot_promote_fbcNN_or_readiness() -> None:
    payload = _read(EVIDENCE)
    assert payload["summary"]["production_qualified"] is False
    assert "single_public_development_identity" in payload["production_blockers"]
    assert "checkpoint_separate_terms_not_verified" in payload["production_blockers"]

    readiness = _read(READINESS)
    gate = next(item for item in readiness["gates"] if item["id"] == "restorer_pack_qualified")
    assert gate["status"] == "NOT_VERIFIED"
    assert "artifact:9502200502" in gate["evidence_refs"]
    assert (
        "artifact-sha256:365251ee8b17dc31099569d328e52439fd6440e869f0ddbe16c4cb4116112842"
        in gate["evidence_refs"]
    )
    assert "commit:666cdbcfbdeee8f20901ccd063a4427d739bd107" in gate["evidence_refs"]
    assert "github-run:33800982565" in gate["evidence_refs"]
    assert "artifact:9916130291" in gate["evidence_refs"]
    assert (
        "artifact-sha256:79b2b0269f982e4ca16d0eb37264f9d5300c767d090127e1500c9feda6926085"
        in gate["evidence_refs"]
    )
    assert "48/48 cases" in gate["detail"]
    assert "eight identities" in gate["detail"]
