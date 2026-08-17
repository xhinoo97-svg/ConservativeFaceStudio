from __future__ import annotations

"""Fail-closed hardening for the V4 MAIN-bridged identity policy."""

from functools import wraps

_INSTALLED = False


def _face_local_identity_bridge_sources(workspace) -> set[int]:
    evidence = workspace.metadata.get("same_canvas_primary_anchor")
    if not isinstance(evidence, dict):
        return set()
    try:
        restored_source = int(evidence.get("restored_source_index", -1))
    except (TypeError, ValueError):
        return set()
    if restored_source != 0:
        return set()
    if evidence.get("identity_bridge_requires_face_local_observed_agreement") is not True:
        return set()
    values = evidence.get("identity_bridge_original_reference_indices")
    if not isinstance(values, list):
        return set()
    result: set[int] = set()
    for value in values:
        try:
            source = int(value)
        except (TypeError, ValueError):
            continue
        if source > 0:
            result.add(source)
    return result


def _require_real_sface_result(result) -> None:
    from app.execution import BlockExecutionError

    details = getattr(result, "details", None)
    if not isinstance(details, dict):
        raise BlockExecutionError(
            "Controllo identità V4 senza evidenza strutturata SFace"
        )
    scores = details.get("scores")
    if not isinstance(scores, list) or not scores:
        raise BlockExecutionError(
            "Controllo identità V4 senza confronti SFace utilizzabili"
        )
    engine = str(details.get("engine", "")).lower()
    if "sface" not in engine:
        raise BlockExecutionError(
            "Controllo identità senza confronto SFace reale: il fallback proxy non è autorità V4"
        )


def install_identity_anchor_v4_hardening() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.identity_anchor_v4_policy as policy

    original_same_canvas = policy._same_canvas_original_sources
    if not getattr(original_same_canvas, "_cfs_v4_face_local_hardened", False):
        @wraps(original_same_canvas)
        def face_local_same_canvas(workspace):
            sources = _face_local_identity_bridge_sources(workspace)
            workspace.metadata["identity_face_local_same_canvas_bridge_original_source_indices"] = sorted(sources)
            return sources

        face_local_same_canvas._cfs_v4_face_local_hardened = True  # type: ignore[attr-defined]
        policy._same_canvas_original_sources = face_local_same_canvas

    original_require = policy._require_identity_result_evidence
    if not getattr(original_require, "_cfs_v4_real_sface_hardened", False):
        @wraps(original_require)
        def require_real_sface(result):
            original_require(result)
            _require_real_sface_result(result)

        require_real_sface._cfs_v4_real_sface_hardened = True  # type: ignore[attr-defined]
        policy._require_identity_result_evidence = require_real_sface

    _INSTALLED = True
