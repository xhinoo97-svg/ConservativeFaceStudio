from __future__ import annotations

"""Do not turn heuristic occlusion proposals into permanent evidence deletion.

For a sparse reference already verified as an exact coordinate-preserving same-canvas
source, the generic dark/chroma/scribble detector is only a proposal. Black padding can
make that proposal extremely noisy near the supported patch and real eyes/hair/shadows
can also be dark. If two coordinate-preserving references overlap and agree almost
exactly on observed pixels, that agreement certifies the reference pair's photometric
canvas. In that narrow case, unresolved proposal-only pixels inside the explicit support
remain original observed evidence instead of being erased by preclean.

Pixels actually repaired by another reference keep the true donor source id. Arbitrary
aligned/different-pose references are never upgraded by this policy.
"""

from functools import wraps
from typing import Any

import cv2
import numpy as np

_INSTALLED = False
_MIN_OVERLAP_PIXELS = 32
_MEDIAN_DIFF_LIMIT = 3.0
_P95_DIFF_LIMIT = 8.0


def _coordinate_slots(workspace, count: int) -> set[int]:
    diagnostics = workspace.metadata.get("same_canvas_partial_alignment_diagnostics")
    runtime_indices = workspace.metadata.get("aligned_reference_source_indices")
    if not isinstance(diagnostics, list) or not isinstance(runtime_indices, list) or len(runtime_indices) != count:
        return set()
    normalized = [int(v) for v in runtime_indices]
    slots: set[int] = set()
    for item in diagnostics:
        if not isinstance(item, dict):
            continue
        coordinate = bool(
            item.get("local_identity_transform", False)
            and item.get("global_transform_required", False) is False
            and str(item.get("method", "")) == "verified-same-canvas-partial"
        )
        if not coordinate:
            continue
        try:
            slot = normalized.index(int(item.get("runtime_reference_index")))
        except (TypeError, ValueError):
            continue
        slots.add(int(slot))
    return slots


def _agreeing_coordinate_slots(refs: list[np.ndarray], supports: list[np.ndarray], candidates: set[int]) -> set[int]:
    """Require at least one almost-exact overlapping coordinate peer per promoted slot."""
    trusted: set[int] = set()
    ordered = sorted(candidates)
    for pos, left_index in enumerate(ordered):
        left = np.asarray(refs[left_index])
        left_support = np.asarray(supports[left_index]) > 0
        for right_index in ordered[pos + 1 :]:
            right = np.asarray(refs[right_index])
            right_support = np.asarray(supports[right_index]) > 0
            overlap = left_support & right_support
            count = int(np.count_nonzero(overlap))
            if count < _MIN_OVERLAP_PIXELS:
                continue
            delta = np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32)), axis=2)[overlap]
            if delta.size == 0:
                continue
            median = float(np.median(delta))
            p95 = float(np.percentile(delta, 95.0))
            if median <= _MEDIAN_DIFF_LIMIT and p95 <= _P95_DIFF_LIMIT:
                trusted.add(left_index)
                trusted.add(right_index)
    return trusted


def install_coordinate_reference_evidence_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.cross_reference_preclean as module

    previous = module.preclean_aligned_references
    if getattr(previous, "_coordinate_reference_evidence_policy", False):
        _INSTALLED = True
        return

    @wraps(previous)
    def preserved(workspace):
        # Capture the actual aligned observations before preclean changes any working copy.
        original_refs = [np.asarray(item).copy() for item in workspace.aligned_references]
        count = len(original_refs)
        if count == 0:
            return previous(workspace)
        shape = workspace.primary.shape[:2]
        supports = [module._support_for_reference(workspace, index, shape) for index in range(count)]
        source_ids = module._source_indices(workspace, count)
        coordinate = _coordinate_slots(workspace, count)
        trusted = _agreeing_coordinate_slots(original_refs, supports, coordinate)

        cleaned, evidence_maps, stats = previous(workspace)
        if not trusted:
            workspace.metadata["coordinate_reference_consensus_trusted_slots"] = []
            workspace.metadata["coordinate_reference_evidence_recovered_pixels"] = 0
            return cleaned, evidence_maps, stats

        recovered_total = 0
        updated_stats: list[module.ReferencePrecleanStats] = []
        for index in range(count):
            evidence = np.asarray(evidence_maps[index]).copy()
            recovered = 0
            if index in trusted:
                support = np.asarray(supports[index]) > 0
                # Zero evidence means preclean's heuristic proposal was unresolved. The
                # working pixel is therefore still the immutable observed pixel from this
                # exact coordinate reference; repaired pixels already have non-zero donor
                # provenance and are deliberately left untouched.
                proposal_only = support & (evidence == 0)
                recovered = int(np.count_nonzero(proposal_only))
                if recovered:
                    evidence[proposal_only] = np.uint16(source_ids[index])
                    evidence_maps[index] = evidence
                    recovered_total += recovered

            old = stats[index]
            unresolved = max(0, int(old.unresolved_pixels) - recovered)
            updated_stats.append(
                module.ReferencePrecleanStats(
                    reference_index=int(old.reference_index),
                    damaged_pixels=int(old.damaged_pixels),
                    repaired_observed_pixels=int(old.repaired_observed_pixels),
                    unresolved_pixels=int(unresolved),
                    donor_sources=tuple(int(v) for v in old.donor_sources),
                )
            )

        workspace.metadata["preclean_reference_evidence_maps"] = [np.asarray(item).copy() for item in evidence_maps]
        damage = module._reference_damage_masks(workspace, count, shape)
        workspace.metadata["preclean_reference_unresolved_masks"] = [
            np.where((damage[index] > 0) & (np.asarray(evidence_maps[index]) == 0), 255, 0).astype(np.uint8)
            for index in range(count)
        ]
        workspace.metadata["preclean_reference_stats"] = [item.__dict__.copy() for item in updated_stats]
        workspace.metadata["coordinate_reference_consensus_trusted_slots"] = sorted(int(v) for v in trusted)
        workspace.metadata["coordinate_reference_evidence_recovered_pixels"] = int(recovered_total)
        workspace.metadata["coordinate_reference_evidence_policy"] = {
            "scope": "verified-coordinate-preserving-partials-only",
            "minimum_overlap_pixels": int(_MIN_OVERLAP_PIXELS),
            "median_difference_limit": float(_MEDIAN_DIFF_LIMIT),
            "p95_difference_limit": float(_P95_DIFF_LIMIT),
            "heuristic_proposal_is_not_final_damage": True,
            "repaired_donor_provenance_preserved": True,
        }
        return cleaned, evidence_maps, updated_stats

    preserved._coordinate_reference_evidence_policy = True  # type: ignore[attr-defined]
    module.preclean_aligned_references = preserved
    _INSTALLED = True
