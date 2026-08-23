from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_preflight_persists_existing_pairwise_sface_matrix() -> None:
    preflight = _text("app/preflight.py")
    assert 'workspace.metadata["preflight_identity_similarity"]' in preflight
    assert '"source": "preflight_existing_sface_embeddings"' in preflight
    assert '"minimum": float(FACE_MODEL_DEFAULTS.sface_same_identity_cosine)' in preflight


def test_v4_hardening_forbids_transitive_component_authority() -> None:
    hardening = _text("app/identity_anchor_v4_hardening.py")
    assert "Newly trusted references never become new" in hardening
    assert "identity_transitive_component_authority_disabled" in hardening
    assert "_direct_identity_authority" in hardening
    assert "value >= minimum" in hardening


def test_candidate_freeze_records_direct_edge_rule_without_changing_threshold() -> None:
    candidate = _text("scripts/freeze_face_domain_guard_v4_candidate.py")
    assert '"identity_firewall_threshold": 0.363' in candidate
    assert '"single_link_component_rule": "ranking-only; never-identity-authority; no-transitive-trust"' in candidate
    assert '"direct_sface_matrix_rule": "reuse-preflight-existing-sface-pairwise-matrix; fixed-authority-direct-edges-only; no-second-biometric-inference"' in candidate
    assert '"same_canvas_rule": "whole-canvas-match-plus-inner-face-peripheral-identity-proof; direct-sface-edge-only"' in candidate
    assert '"final_holdout_manifests_unchanged_since_introduction": True' in candidate
