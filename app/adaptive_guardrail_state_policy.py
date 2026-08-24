from __future__ import annotations

"""Include adaptive-restoration metadata in outer automatic rollback snapshots.

The adaptive cascade keeps masks/reports that influence later blocks. If an outer
identity/quality guardrail rejects Block 8, those fields must roll back together with
the image and provenance map; otherwise Block 9 can inherit a stale 'already fixed'
mask for pixels that were actually reverted.
"""

_INSTALLED = False

_ADAPTIVE_KEYS = (
    "protected_region_mask",
    "adaptive_blur_classification",
    "adaptive_severity_map",
    "adaptive_severity_counts",
    "adaptive_severity_from_immutable_main",
    "adaptive_restoration_stage",
    "adaptive_restoration_stage_mask",
    "adaptive_restoration_reports",
    "adaptive_restoration_remaining_mask",
    "adaptive_preexisting_observed_protected_pixels",
    "adaptive_preexisting_observed_protection",
)


def install_adaptive_guardrail_state_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from app.automatic import AutomaticPipelineRunner

    existing = tuple(AutomaticPipelineRunner._GUARDRAIL_METADATA_KEYS)
    AutomaticPipelineRunner._GUARDRAIL_METADATA_KEYS = tuple(
        dict.fromkeys((*existing, *_ADAPTIVE_KEYS))
    )
    _INSTALLED = True
