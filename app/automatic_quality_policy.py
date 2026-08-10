from __future__ import annotations

from typing import Any

from app.execution import ExecutionResult


_INSTALLED = False


def install_automatic_quality_policy() -> None:
    """Make the automatic ENHANCE block real without making it aggressive.

    The automatic runner historically passed ``blend=0.0`` to ENHANCE.  That made
    the block a no-op even though it was exported as one of the 13 processing stages.
    In automatic strict mode, zero is now interpreted as "auto conservative": a small
    CLAHE blend is applied and the existing per-block identity guardrail can still
    roll it back if it changes the face too much.  Manual callers that pass a
    positive blend keep their requested value unchanged.
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
        if automatic_conservative:
            # Deliberately modest: enough to make the stage functional while leaving
            # identity/skin geometry to the observed data rather than contrast priors.
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
                "identity_guardrail_required": True,
            }
        )
        return ExecutionResult(result.block, result.image, details)

    StrictBlockExecutor._enhance = patched_enhance
    _INSTALLED = True
