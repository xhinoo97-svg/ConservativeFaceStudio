from __future__ import annotations

"""Fail-closed hardening for the V4 MAIN-bridged identity policy.

The preflight identity component is useful for ranking, but it is single-link: A-B and
B-C can form one component even when A-C is below the SFace threshold. V4 identity
authority therefore never propagates transitively through that component. It uses only
direct SFace evidence anchored to MAIN source 0 or to an independently face-local
same-canvas bridge source. Existing current-stage direct SFace flags remain usable when
no valid preflight matrix exists; a missing historical `engine` label is not itself proof
of a proxy, while an explicitly non-SFace engine is rejected.
"""

from functools import wraps
import math

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


def _preflight_direct_sface_edges(workspace) -> tuple[dict[int, int], list[list[float]], float] | None:
    """Parse the already-computed preflight SFace matrix; never run another model."""
    from app.pretrained_values import FACE_MODEL_DEFAULTS

    payload = workspace.metadata.get("preflight_identity_similarity")
    if not isinstance(payload, dict):
        return None
    if payload.get("source") != "preflight_existing_sface_embeddings":
        return None
    raw_sources = payload.get("source_indices")
    raw_matrix = payload.get("matrix")
    if not isinstance(raw_sources, list) or not isinstance(raw_matrix, list):
        return None
    try:
        sources = [int(value) for value in raw_sources]
    except (TypeError, ValueError):
        return None
    max_source = len(workspace.references)
    if (
        not sources
        or len(set(sources)) != len(sources)
        or len(raw_matrix) != len(sources)
        or any(source < 0 or source > max_source for source in sources)
    ):
        return None

    matrix: list[list[float]] = []
    try:
        for row in raw_matrix:
            if not isinstance(row, list) or len(row) != len(sources):
                return None
            values = [float(value) for value in row]
            if any(not math.isfinite(value) or value < -1.001 or value > 1.001 for value in values):
                return None
            matrix.append(values)
        recorded_minimum = float(payload.get("minimum", FACE_MODEL_DEFAULTS.sface_same_identity_cosine))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(recorded_minimum) or recorded_minimum < -1.0 or recorded_minimum > 1.0:
        return None

    for index in range(len(sources)):
        if abs(matrix[index][index] - 1.0) > 1e-5:
            return None
        for other in range(index + 1, len(sources)):
            if abs(matrix[index][other] - matrix[other][index]) > 1e-5:
                return None

    minimum = max(float(FACE_MODEL_DEFAULTS.sface_same_identity_cosine), recorded_minimum)
    positions = {source: index for index, source in enumerate(sources)}
    return positions, matrix, minimum


def _direct_identity_authority(workspace) -> dict[int, tuple[int, ...]]:
    """Return target source -> fixed authority anchors with a direct SFace edge.

    Authorities are fixed before traversal: MAIN (if it has an embedding) plus exact
    face-local same-canvas bridge sources. Newly trusted references never become new
    anchors, preventing single-link/transitive identity propagation.
    """
    parsed = _preflight_direct_sface_edges(workspace)
    if parsed is None:
        workspace.metadata["identity_direct_sface_matrix_valid"] = False
        workspace.metadata["identity_direct_sface_authority"] = {}
        return {}
    positions, matrix, minimum = parsed
    same_canvas = _face_local_identity_bridge_sources(workspace)
    anchors = ({0} if 0 in positions else set()) | {source for source in same_canvas if source in positions}
    authority: dict[int, tuple[int, ...]] = {}
    for source, source_position in positions.items():
        if source <= 0:
            continue
        linked: list[int] = []
        for anchor in sorted(anchors):
            if anchor == source:
                continue
            value = matrix[source_position][positions[anchor]]
            if value >= minimum:
                linked.append(anchor)
        if linked:
            authority[source] = tuple(linked)

    workspace.metadata["identity_direct_sface_matrix_valid"] = True
    workspace.metadata["identity_direct_sface_threshold"] = float(minimum)
    workspace.metadata["identity_direct_sface_fixed_anchor_original_source_indices"] = sorted(anchors)
    workspace.metadata["identity_direct_sface_authority"] = {
        str(source): list(anchors_for_source)
        for source, anchors_for_source in sorted(authority.items())
    }
    return authority


def _runtime_order(workspace, reference_count: int) -> list[int]:
    raw = workspace.metadata.get("runtime_source_order")
    if isinstance(raw, list) and len(raw) == reference_count + 1:
        try:
            return [int(value) for value in raw]
        except (TypeError, ValueError):
            pass
    return list(range(reference_count + 1))


def _current_direct_sface_sources(workspace, reference_count: int) -> set[int]:
    """Return current-stage whole-face SFace-positive sources.

    A current `reference_identity_verified=True` plus a usable numeric score is direct
    evidence produced for the current runtime slot. It is mapped through
    `runtime_source_order`, so reordering cannot change the original source identity.
    This fallback is used when no valid preflight matrix is available; it does not
    create transitive trust.
    """
    flags = workspace.metadata.get("reference_identity_verified")
    scores = workspace.metadata.get("reference_identity_scores")
    if not isinstance(flags, list) or len(flags) != reference_count:
        return set()
    if not isinstance(scores, list) or len(scores) != reference_count:
        return set()
    order = _runtime_order(workspace, reference_count)
    result: set[int] = set()
    for index, flag in enumerate(flags):
        if not bool(flag) or index + 1 >= len(order):
            continue
        score = scores[index]
        if score is None:
            continue
        try:
            value = float(score)
            source = int(order[index + 1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and source > 0:
            result.add(source)
    return result


def _harden_bridge_result(
    workspace,
    original_bridge,
) -> tuple[list[bool], list[str], set[int]]:
    count = len(workspace.references)
    before_flags_raw = workspace.metadata.get("reference_identity_verified")
    before_flags = (
        [bool(value) for value in before_flags_raw]
        if isinstance(before_flags_raw, list) and len(before_flags_raw) == count
        else [False] * count
    )
    before_reasons_raw = workspace.metadata.get("reference_identity_reasons")
    before_reasons = (
        [str(value) for value in before_reasons_raw]
        if isinstance(before_reasons_raw, list) and len(before_reasons_raw) == count
        else ["direct_sface" if value else "rejected" for value in before_flags]
    )

    flags, reasons, _trusted = original_bridge(workspace)
    authority = _direct_identity_authority(workspace)
    parsed = _preflight_direct_sface_edges(workspace)
    same_canvas = _face_local_identity_bridge_sources(workspace)
    current_direct = _current_direct_sface_sources(workspace, count)
    order = _runtime_order(workspace, count)

    for index in range(count):
        try:
            source = int(order[index + 1])
        except (TypeError, ValueError, IndexError):
            flags[index] = False
            reasons[index] = "rejected_invalid_source_mapping"
            continue

        score_values = workspace.metadata.get("reference_identity_scores")
        current_score = score_values[index] if isinstance(score_values, list) and len(score_values) == count else None

        # A face-local same-canvas source is a global anchor only if it also has a
        # whole-face identity observation. Sparse/partial sheets remain component-local.
        if source in same_canvas and flags[index] and current_score is not None:
            reasons[index] = "verified_face_local_same_canvas_main_bridge"
            continue

        if before_flags[index] and before_reasons[index] == "direct_sface" and current_score is not None:
            flags[index] = True
            reasons[index] = "direct_sface"
            continue

        linked = authority.get(source, ())
        if linked and current_score is not None:
            flags[index] = True
            if 0 in linked:
                reasons[index] = "direct_main_sface_bridge"
            else:
                reasons[index] = "same_canvas_direct_sface_bridge"
            continue

        # If preflight has no valid matrix, preserve only direct current-stage SFace
        # evidence. This is not a reference-cluster rescue: each source has its own
        # current score/flag and is mapped to its original source id.
        if parsed is None and source in current_direct:
            flags[index] = True
            if reasons[index] in {"rejected", ""}:
                reasons[index] = "direct_sface"
            continue

        if flags[index] or before_flags[index] or reasons[index] in {
            "same_canvas_bridged_cross_reference_cluster",
            "main_bridged_cross_reference_cluster",
        }:
            flags[index] = False
            reasons[index] = "rejected_transitive_component_only"

    trusted = {
        int(order[index + 1])
        for index, flag in enumerate(flags)
        if flag
        and index + 1 < len(order)
        and isinstance(workspace.metadata.get("reference_identity_scores"), list)
        and len(workspace.metadata["reference_identity_scores"]) == count
        and workspace.metadata["reference_identity_scores"][index] is not None
    }
    workspace.metadata["reference_identity_verified"] = flags
    workspace.metadata["reference_identity_reasons"] = reasons
    workspace.metadata["identity_trusted_original_source_indices"] = sorted(trusted)
    workspace.metadata["identity_transitive_component_authority_disabled"] = True
    workspace.metadata["identity_v4_flags_hardened"] = True
    return flags, reasons, trusted


def _harden_trusted_sources(workspace, reference_count: int, original_trusted) -> set[int]:
    """Return global identity anchors with explicit full-face evidence only."""
    flags = workspace.metadata.get("reference_identity_verified")
    scores = workspace.metadata.get("reference_identity_scores")
    reasons = workspace.metadata.get("reference_identity_reasons")
    order = _runtime_order(workspace, reference_count)
    parsed = _preflight_direct_sface_edges(workspace)
    authority = _direct_identity_authority(workspace)
    same_canvas = _face_local_identity_bridge_sources(workspace)
    hardened = workspace.metadata.get("identity_v4_flags_hardened") is True

    score_values = scores if isinstance(scores, list) and len(scores) == reference_count else None
    reason_values = reasons if isinstance(reasons, list) and len(reasons) == reference_count else ["rejected"] * reference_count

    trusted: set[int] = set()

    # When current LANDMARKS/SFace flags exist, they describe the current runtime slots.
    # A usable current score is required, which prevents a partial same-canvas sheet from
    # becoming a global anchor. If a valid preflight matrix exists and flags have not yet
    # been V4-hardened, only fixed direct authority or an explicit direct_sface reason may
    # survive. Without a valid matrix, current per-source SFace flags are the best direct
    # evidence available and are preserved.
    if isinstance(flags, list) and len(flags) == reference_count and score_values is not None:
        for index, flag in enumerate(flags):
            if not bool(flag) or score_values[index] is None or index + 1 >= len(order):
                continue
            try:
                source = int(order[index + 1])
            except (TypeError, ValueError):
                continue
            if source <= 0:
                continue
            if hardened:
                trusted.add(source)
            elif parsed is None:
                trusted.add(source)
            elif str(reason_values[index]) == "direct_sface" or source in authority:
                trusted.add(source)

    else:
        # Before current SFace flags exist, use only the fixed preflight direct graph.
        trusted.update(source for source in authority if source > 0)
        # A strict face-local same-canvas source may itself be a global anchor only when
        # the preflight matrix proves that a whole-face embedding existed for that source.
        if parsed is not None:
            positions = parsed[0]
            trusted.update(source for source in same_canvas if source in positions)

    workspace.metadata["identity_transitive_component_authority_disabled"] = True
    workspace.metadata["identity_pre_landmarks_direct_trusted_original_source_indices"] = sorted(trusted)
    return trusted


def _require_real_sface_result(result) -> None:
    from app.execution import BlockExecutionError

    details = getattr(result, "details", None)
    if not isinstance(details, dict):
        raise BlockExecutionError("Controllo identità V4 senza evidenza strutturata SFace")
    scores = details.get("scores")
    if not isinstance(scores, list) or not scores:
        raise BlockExecutionError("Controllo identità V4 senza confronti SFace utilizzabili")
    try:
        numeric = [float(value) for value in scores]
    except (TypeError, ValueError):
        raise BlockExecutionError("Controllo identità V4 con score biometrico non valido")
    if any(not math.isfinite(value) for value in numeric):
        raise BlockExecutionError("Controllo identità V4 con score biometrico non finito")

    # Historical valid handlers did not always emit an `engine` field. Missing metadata
    # alone is not proof that the scores came from a proxy. But if an engine is explicitly
    # declared, it must identify SFace; explicit histogram/proxy engines fail closed.
    if "engine" in details:
        engine = str(details.get("engine", "")).strip().lower()
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

    original_bridge = policy._bridge_reference_identity
    if not getattr(original_bridge, "_cfs_v4_direct_edge_hardened", False):
        @wraps(original_bridge)
        def direct_edge_bridge(workspace):
            return _harden_bridge_result(workspace, original_bridge)

        direct_edge_bridge._cfs_v4_direct_edge_hardened = True  # type: ignore[attr-defined]
        policy._bridge_reference_identity = direct_edge_bridge

    original_trusted = policy._trusted_identity_source_indices
    if not getattr(original_trusted, "_cfs_v4_direct_edge_hardened", False):
        @wraps(original_trusted)
        def direct_edge_trusted(workspace, reference_count: int):
            return _harden_trusted_sources(workspace, reference_count, original_trusted)

        direct_edge_trusted._cfs_v4_direct_edge_hardened = True  # type: ignore[attr-defined]
        policy._trusted_identity_source_indices = direct_edge_trusted

    original_require = policy._require_identity_result_evidence
    if not getattr(original_require, "_cfs_v4_real_sface_hardened", False):
        @wraps(original_require)
        def require_real_sface(result):
            original_require(result)
            _require_real_sface_result(result)

        require_real_sface._cfs_v4_real_sface_hardened = True  # type: ignore[attr-defined]
        policy._require_identity_result_evidence = require_real_sface

    _INSTALLED = True
