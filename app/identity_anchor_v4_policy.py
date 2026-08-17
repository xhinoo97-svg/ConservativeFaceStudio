from __future__ import annotations

"""V4 identity-anchor policy for severely damaged MAIN photographs.

Cross-reference agreement alone is never identity authority. A reference may extend
whole-face identity trust only when the imported MAIN is bridged by either the frozen
SFace rule already used by V2 or by the stricter pixel-level same-canvas verifier.
The SFace threshold is unchanged; this layer only preserves stronger evidence that
was previously discarded by the preflight-only firewall.
"""

from functools import wraps
from typing import Any

import numpy as np


_INSTALLED = False
POLICY_NAME = "main-bridged-identity-anchor-v4"


def _runtime_order(workspace, reference_count: int) -> list[int]:
    raw = workspace.metadata.get("runtime_source_order")
    if isinstance(raw, list) and len(raw) == reference_count + 1:
        try:
            return [int(value) for value in raw]
        except (TypeError, ValueError):
            pass
    return list(range(reference_count + 1))


def _same_canvas_original_sources(workspace) -> set[int]:
    """Return references already proven to share the imported MAIN canvas.

    `applied` is deliberately not required. The primary-anchor policy records an
    `applied=False` decision when MAIN was already source 0, while the matched source
    list still represents a successful pixel-level same-canvas verification.
    """
    evidence = workspace.metadata.get("same_canvas_primary_anchor")
    if not isinstance(evidence, dict):
        return set()
    try:
        restored_source = int(evidence.get("restored_source_index", -1))
    except (TypeError, ValueError):
        return set()
    if restored_source != 0:
        return set()
    values = evidence.get("matched_original_reference_indices")
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


def _preflight_candidate_map(workspace) -> dict[int, dict[str, Any]]:
    values = workspace.metadata.get("preflight_candidates")
    if not isinstance(values, list):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict):
            continue
        try:
            source = int(item.get("source_index"))
        except (TypeError, ValueError):
            continue
        result[source] = item
    return result


def _preflight_accepted_sources(workspace) -> set[int]:
    return {
        source
        for source, item in _preflight_candidate_map(workspace).items()
        if bool(item.get("accepted_identity", False))
    }


def _effective_identity_eligibility(
    workspace,
    eligibility: dict[int, str],
    *,
    accepted_value: str,
) -> dict[int, str]:
    """Upgrade only exact pixel-verified same-canvas sources in the V2 firewall."""
    result = dict(eligibility)
    same_canvas = _same_canvas_original_sources(workspace)
    for source in same_canvas:
        result[source] = accepted_value
    workspace.metadata["identity_firewall_same_canvas_override_original_source_indices"] = sorted(same_canvas)
    workspace.metadata["identity_anchor_policy"] = POLICY_NAME
    return result


def _bridge_reference_identity(workspace) -> tuple[list[bool], list[str], set[int]]:
    """Extend LANDMARKS identity flags only through a direct MAIN bridge.

    A same-canvas source with a current full-face SFace observation is trusted directly.
    If any same-canvas source belongs to the preflight accepted identity component, that
    component may extend trust to its other currently face-detectable references. A
    reference-only component with no MAIN/same-canvas bridge is never promoted.
    """
    count = len(workspace.references)
    existing_flags = workspace.metadata.get("reference_identity_verified")
    flags = [False] * count
    if isinstance(existing_flags, list) and len(existing_flags) == count:
        flags = [bool(value) for value in existing_flags]

    existing_reasons = workspace.metadata.get("reference_identity_reasons")
    reasons = ["rejected"] * count
    if isinstance(existing_reasons, list) and len(existing_reasons) == count:
        reasons = [str(value) for value in existing_reasons]
    else:
        reasons = ["direct_sface" if flag else "rejected" for flag in flags]

    scores_raw = workspace.metadata.get("reference_identity_scores")
    scores = scores_raw if isinstance(scores_raw, list) and len(scores_raw) == count else [None] * count
    order = _runtime_order(workspace, count)
    same_canvas = _same_canvas_original_sources(workspace)
    accepted = _preflight_accepted_sources(workspace)
    cluster_bridged = bool(same_canvas & accepted)

    for index in range(count):
        source = int(order[index + 1])
        score = scores[index]
        # `score is not None` distinguishes a whole-face SFace observation from a
        # sparse/partial sheet. Same-canvas partials remain usable only through the
        # existing local geometry/provenance path, never as global identity anchors.
        if score is None:
            continue
        if source in same_canvas:
            flags[index] = True
            reasons[index] = "verified_same_canvas_main_bridge"
        elif cluster_bridged and source in accepted:
            flags[index] = True
            reasons[index] = "same_canvas_bridged_cross_reference_cluster"

    trusted = {
        int(order[index + 1])
        for index, flag in enumerate(flags)
        if flag and index + 1 < len(order)
    }
    workspace.metadata["reference_identity_verified"] = flags
    workspace.metadata["reference_identity_reasons"] = reasons
    workspace.metadata["identity_trusted_original_source_indices"] = sorted(trusted)
    workspace.metadata["identity_same_canvas_bridge_original_source_indices"] = sorted(same_canvas)
    workspace.metadata["identity_anchor_policy"] = POLICY_NAME
    return flags, reasons, trusted


def _trusted_identity_source_indices(workspace, reference_count: int) -> set[int]:
    """Return whole-face identity anchors without trusting a reference-only cluster."""
    order = _runtime_order(workspace, reference_count)
    flags = workspace.metadata.get("reference_identity_verified")
    scores = workspace.metadata.get("reference_identity_scores")
    if isinstance(flags, list) and len(flags) == reference_count:
        score_values = scores if isinstance(scores, list) and len(scores) == reference_count else [None] * reference_count
        return {
            int(order[index + 1])
            for index, flag in enumerate(flags)
            if bool(flag) and score_values[index] is not None and index + 1 < len(order)
        }

    candidates = _preflight_candidate_map(workspace)
    accepted = {source for source, item in candidates.items() if bool(item.get("accepted_identity", False))}
    same_canvas = _same_canvas_original_sources(workspace)
    main_bridged = 0 in accepted
    same_canvas_cluster_bridged = bool(accepted & same_canvas)

    trusted: set[int] = set()
    if main_bridged or same_canvas_cluster_bridged:
        trusted.update(source for source in accepted if source > 0)

    # Exact same-canvas sources rejected from the selected component are still valid
    # whole-face anchors only when preflight actually produced an identity embedding.
    for source in same_canvas:
        candidate = candidates.get(source)
        if isinstance(candidate, dict) and candidate.get("identity_embedding_available") is True:
            trusted.add(source)
    return trusted


def _trusted_raw_reference_positions(workspace) -> tuple[list[int], list[int]]:
    count = len(workspace.references)
    trusted = _trusted_identity_source_indices(workspace, count)
    order = _runtime_order(workspace, count)
    positions: list[int] = []
    sources: list[int] = []
    for index in range(count):
        source = int(order[index + 1])
        if source in trusted:
            positions.append(index)
            sources.append(source)
    return positions, sources


def _install_v2_same_canvas_override() -> None:
    import app.face_domain_guard_v2_policy as v2

    original = v2._identity_eligibility_by_source
    if getattr(original, "_cfs_v4_same_canvas_override", False):
        return

    @wraps(original)
    def effective_eligibility(workspace):
        return _effective_identity_eligibility(
            workspace,
            original(workspace),
            accepted_value=v2.IDENTITY_ACCEPTED,
        )

    effective_eligibility._cfs_v4_same_canvas_override = True  # type: ignore[attr-defined]
    v2._identity_eligibility_by_source = effective_eligibility


def _install_handler_bridge() -> None:
    import app.pretrained_face_handlers as handlers

    original_install = handlers.install_pretrained_face_handlers
    if getattr(original_install, "_cfs_v4_identity_anchor", False):
        return

    @wraps(original_install)
    def bridged_install(executor, model_paths):
        original_install(executor, model_paths)
        from app.execution import ExecutionResult
        from app.pipeline import BlockKind

        landmarks = executor._handlers.get(BlockKind.LANDMARKS)
        if landmarks is not None and not getattr(landmarks, "_cfs_v4_identity_anchor", False):
            @wraps(landmarks)
            def bridged_landmarks(block, parameters):
                result = landmarks(block, parameters)
                flags, reasons, trusted = _bridge_reference_identity(executor.workspace)
                details = dict(result.details)
                details["reference_identity_verified"] = int(sum(flags))
                details["reference_identity_reasons"] = reasons
                details["identity_trusted_original_source_indices"] = sorted(trusted)
                details["identity_same_canvas_bridge_original_source_indices"] = sorted(
                    _same_canvas_original_sources(executor.workspace)
                )
                details["identity_anchor_policy"] = POLICY_NAME
                return ExecutionResult(result.block, result.image, details)

            bridged_landmarks._cfs_v4_identity_anchor = True  # type: ignore[attr-defined]
            executor._handlers[BlockKind.LANDMARKS] = bridged_landmarks

        identity = executor._handlers.get(BlockKind.IDENTITY_CHECK)
        if identity is not None and not getattr(identity, "_cfs_v4_identity_anchor", False):
            @wraps(identity)
            def trusted_identity_check(block, parameters):
                workspace = executor.workspace
                original_references = list(workspace.references)
                positions, sources = _trusted_raw_reference_positions(workspace)
                workspace.references = [original_references[index] for index in positions]
                try:
                    result = identity(block, parameters)
                finally:
                    workspace.references = original_references
                details = dict(result.details)
                details["identity_anchor_policy"] = POLICY_NAME
                details["identity_trusted_original_source_indices"] = sources
                details["identity_raw_reference_count"] = len(original_references)
                details["identity_trusted_reference_count"] = len(positions)
                details["identity_excluded_untrusted_reference_count"] = len(original_references) - len(positions)
                return ExecutionResult(result.block, result.image, details)

            trusted_identity_check._cfs_v4_identity_anchor = True  # type: ignore[attr-defined]
            executor._handlers[BlockKind.IDENTITY_CHECK] = trusted_identity_check

    bridged_install._cfs_v4_identity_anchor = True  # type: ignore[attr-defined]
    handlers.install_pretrained_face_handlers = bridged_install


def _install_global_anchor_policy() -> None:
    import app.automatic as automatic

    original = automatic.AutomaticPipelineRunner._global_identity_anchors
    if getattr(original, "_cfs_v4_identity_anchor", False):
        return

    @wraps(original)
    def global_identity_anchors(self):
        workspace = self.executor.workspace
        references = list(workspace.references)
        trusted = _trusted_identity_source_indices(workspace, len(references))
        order = _runtime_order(workspace, len(references))
        anchors = [
            reference
            for index, reference in enumerate(references)
            if index + 1 < len(order) and int(order[index + 1]) in trusted
        ]
        workspace.metadata["identity_global_anchor_original_source_indices"] = sorted(trusted)
        workspace.metadata["identity_anchor_policy"] = POLICY_NAME
        return anchors or [self._original_anchor]

    global_identity_anchors._cfs_v4_identity_anchor = True  # type: ignore[attr-defined]
    automatic.AutomaticPipelineRunner._global_identity_anchors = global_identity_anchors


def install_identity_anchor_v4_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_v2_same_canvas_override()
    _install_handler_bridge()
    _install_global_anchor_policy()
    _INSTALLED = True
