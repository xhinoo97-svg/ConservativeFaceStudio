from __future__ import annotations

from typing import Any

from app.execution import ExecutionResult


_INSTALLED = False


def install_automatic_quality_policy() -> None:
    """Make automatic ENHANCE useful while preserving reference-driven evidence.

    ``blend<=0`` means automatic conservative selection. Single-image cases retain a
    modest CLAHE blend. When real references are available, global contrast remapping
    is counterproductive: later blocks can reconstruct the damaged region from observed
    donor pixels, while a global CLAHE changes already-correct skin/background and the
    donors no longer match photometrically. In that route ENHANCE therefore executes a
    deliberate preserve decision rather than altering pixels merely to make the stage
    visibly active. Manual positive blends remain unchanged.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from app.strict_execution import StrictBlockExecutor

    original = StrictBlockExecutor._enhance

    def patched_enhance(self, block, parameters: dict[str, Any]) -> ExecutionResult:
        p = dict(parameters)
        requested = float(p.get("blend", 0.2))
        automatic_conservative = requested <= 0.0
        preserve_reference_evidence = automatic_conservative and bool(self.workspace.references)

        if automatic_conservative:
            if preserve_reference_evidence:
                p["blend"] = 0.0
            else:
                p["blend"] = 0.12
                p.setdefault("clip_limit", 1.45)

        result = original(self, block, p)
        if not automatic_conservative:
            return result

        details = dict(result.details)
        details.update(
            {
                "automatic_conservative": True,
                "requested_blend": requested,
                "effective_blend": float(p["blend"]),
                "identity_guardrail_required": not preserve_reference_evidence,
                "decision": (
                    "preserve_observed_multi_reference_photometry"
                    if preserve_reference_evidence
                    else "mild_single_image_local_contrast"
                ),
                "reference_evidence_preserved": bool(preserve_reference_evidence),
            }
        )
        return ExecutionResult(result.block, result.image, details)

    StrictBlockExecutor._enhance = patched_enhance
    _INSTALLED = True
