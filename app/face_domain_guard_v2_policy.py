from __future__ import annotations

"""Final release identity firewall and narrow face-damage preservation policy.

This policy does not add identity inference.  It records whether the *existing*
preflight YuNet/SFace observation produced an embedding and preserves that
per-source result through alignment.  A source explicitly rejected by the
preflight identity cluster can never recover observed-donor eligibility merely
because geometric alignment succeeds.  Sources without global identity evidence
remain eligible only through the existing strict partial/component path.
"""

from functools import wraps
from typing import Any

import numpy as np


_INSTALLED = False
IDENTITY_ACCEPTED = "IDENTITY_ACCEPTED"
IDENTITY_REJECTED = "IDENTITY_REJECTED"
PARTIAL_IDENTITY_UNKNOWN = "PARTIAL_IDENTITY_UNKNOWN"


def _candidate_status(item: dict[str, Any]) -> str:
    value = str(item.get("identity_eligibility", ""))
    if value in {IDENTITY_ACCEPTED, IDENTITY_REJECTED, PARTIAL_IDENTITY_UNKNOWN}:
        return value
    available = item.get("identity_embedding_available")
    if available is True:
        return IDENTITY_ACCEPTED if bool(item.get("accepted_identity", False)) else IDENTITY_REJECTED
    return PARTIAL_IDENTITY_UNKNOWN


def _identity_eligibility_by_source(workspace) -> dict[int, str]:
    result: dict[int, str] = {}
    candidates = workspace.metadata.get("preflight_candidates")
    if not isinstance(candidates, list):
        return result
    for item in candidates:
        if not isinstance(item, dict):
            continue
        try:
            source_index = int(item.get("source_index"))
        except (TypeError, ValueError):
            continue
        result[source_index] = _candidate_status(item)
    return result


def _filter_aligned_references(workspace) -> list[int]:
    """Remove explicitly rejected original photos before any donor use downstream."""
    refs = list(workspace.aligned_references)
    original_indices = workspace.metadata.get("aligned_reference_original_source_indices")
    if not isinstance(original_indices, list) or len(original_indices) != len(refs):
        return []
    eligibility = _identity_eligibility_by_source(workspace)
    rejected_slots = [
        index
        for index, original in enumerate(original_indices)
        if eligibility.get(int(original), PARTIAL_IDENTITY_UNKNOWN) == IDENTITY_REJECTED
    ]
    if not rejected_slots:
        return []

    rejected_set = set(rejected_slots)
    rejected_original = {int(original_indices[index]) for index in rejected_slots}
    keep = [index for index in range(len(refs)) if index not in rejected_set]
    workspace.aligned_references = [refs[index] for index in keep]

    parallel_keys = (
        "aligned_reference_source_indices",
        "aligned_reference_original_source_indices",
        "aligned_reference_support_masks",
        "aligned_reference_detail_reliability_maps",
        "aligned_reference_identity_scores",
        "aligned_reference_identity_verified",
        "aligned_reference_partial_geometry_verified",
    )
    for key in parallel_keys:
        values = workspace.metadata.get(key)
        if isinstance(values, list) and len(values) == len(refs):
            workspace.metadata[key] = [values[index] for index in keep]

    bank = workspace.metadata.get("component_reference_bank")
    if isinstance(bank, dict):
        filtered_bank: dict[str, list[Any]] = {}
        for name, entries in bank.items():
            if not isinstance(entries, list):
                continue
            kept_entries: list[Any] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    kept_entries.append(entry)
                    continue
                try:
                    source_index = int(entry.get("source_index", -1))
                except (TypeError, ValueError):
                    kept_entries.append(entry)
                    continue
                if source_index not in rejected_original:
                    kept_entries.append(entry)
            filtered_bank[str(name)] = kept_entries
        workspace.metadata["component_reference_bank"] = filtered_bank

    workspace.metadata["identity_firewall_rejected_original_source_indices"] = sorted(rejected_original)
    workspace.metadata["identity_firewall_policy"] = "preflight-per-source-v2"
    return sorted(rejected_original)


def _install_preflight_identity_audit() -> None:
    import app.preflight as preflight

    base_engine = preflight.OpenCVZooFaceEngine
    if getattr(base_engine, "_cfs_v2_identity_audit", False):
        return

    class RecordingPreflightFaceEngine(base_engine):
        _cfs_v2_identity_audit = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._preflight_embedding_available: list[bool] = []

        def analyze(self, image):
            try:
                observation = super().analyze(image)
            except Exception:
                self._preflight_embedding_available.append(False)
                raise
            self._preflight_embedding_available.append(observation.embedding is not None)
            return observation

    preflight.OpenCVZooFaceEngine = RecordingPreflightFaceEngine
    original_preprocess = preflight.preprocess_and_select_front_base

    @wraps(original_preprocess)
    def audited_preprocess(workspace, model_paths):
        result = original_preprocess(workspace, model_paths)
        candidates = workspace.metadata.get("preflight_candidates")
        backend = workspace.metadata.get("_identity_backend")
        availability = getattr(backend, "_preflight_embedding_available", None)
        if not isinstance(candidates, list) or not isinstance(availability, list):
            return result
        if len(availability) < len(candidates):
            return result

        eligibility: dict[str, str] = {}
        for index, item in enumerate(candidates):
            if not isinstance(item, dict):
                continue
            available = bool(availability[index])
            item["identity_embedding_available"] = available
            if available:
                status = IDENTITY_ACCEPTED if bool(item.get("accepted_identity", False)) else IDENTITY_REJECTED
            else:
                status = PARTIAL_IDENTITY_UNKNOWN
            item["identity_eligibility"] = status
            try:
                eligibility[str(int(item.get("source_index")))] = status
            except (TypeError, ValueError):
                pass
        workspace.metadata["preflight_identity_eligibility"] = eligibility
        workspace.metadata["preflight_identity_eligibility_policy"] = "same-inference-audit-v2"
        return result

    preflight.preprocess_and_select_front_base = audited_preprocess


def _install_alignment_firewall() -> None:
    import app.pretrained_face_handlers as handlers

    original_install = handlers.install_pretrained_face_handlers
    if getattr(original_install, "_cfs_v2_identity_firewall", False):
        return

    @wraps(original_install)
    def guarded_install(executor, model_paths):
        original_install(executor, model_paths)
        from app.pipeline import BlockKind
        from app.execution import ExecutionResult

        align = executor._handlers.get(BlockKind.ALIGN)
        if align is None or getattr(align, "_cfs_v2_identity_firewall", False):
            return

        @wraps(align)
        def guarded_align(block, parameters):
            result = align(block, parameters)
            rejected = _filter_aligned_references(executor.workspace)
            if not rejected:
                return result
            details = dict(result.details)
            details["aligned"] = len(executor.workspace.aligned_references)
            details["identity_firewall_policy"] = "preflight-per-source-v2"
            details["identity_firewall_rejected_original_source_indices"] = rejected
            details["preflight_identity_rejected"] = len(rejected)
            details["rejected_identity"] = int(details.get("rejected_identity", 0)) + len(rejected)
            return ExecutionResult(result.block, result.image, details)

        guarded_align._cfs_v2_identity_firewall = True  # type: ignore[attr-defined]
        executor._handlers[BlockKind.ALIGN] = guarded_align

    guarded_install._cfs_v2_identity_firewall = True  # type: ignore[attr-defined]
    handlers.install_pretrained_face_handlers = guarded_install


def _install_observed_donor_firewall_and_preservation() -> None:
    import app.observed_target_repair_runtime as runtime
    from app.same_canvas_seed_precision_policy import precise_same_canvas_damage_seed

    original_trusted = runtime._trusted_slots
    if not getattr(original_trusted, "_cfs_v2_identity_firewall", False):
        @wraps(original_trusted)
        def trusted_slots(workspace, count: int):
            flags = list(original_trusted(workspace, count))
            eligibility = _identity_eligibility_by_source(workspace)
            original_indices = runtime._aligned_original_indices(workspace, count)
            for index in range(min(len(flags), len(original_indices))):
                if eligibility.get(int(original_indices[index]), PARTIAL_IDENTITY_UNKNOWN) == IDENTITY_REJECTED:
                    flags[index] = False
            return flags

        trusted_slots._cfs_v2_identity_firewall = True  # type: ignore[attr-defined]
        runtime._trusted_slots = trusted_slots

    # Reuse the already-frozen narrowest authoritative damage seed.  This avoids
    # re-expanding a refined inpaint/reference-consensus target with the broad raw
    # preflight proposal, while preserving the existing fallback chain when no refined
    # evidence exists.
    runtime._target_mask = precise_same_canvas_damage_seed


def install_face_domain_guard_v2_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_preflight_identity_audit()
    _install_alignment_firewall()
    _install_observed_donor_firewall_and_preservation()
    _INSTALLED = True
