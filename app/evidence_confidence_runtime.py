from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

from app.evidence_confidence import compute_evidence_confidence
from app.evidence_accounting import reconcile_evidence_accounting
from app.execution import ExecutionResult
from app.pipeline import BlockKind

_INSTALLED = False


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _source_counts(workspace) -> dict[str, int]:
    provenance = workspace.provenance_map
    if not isinstance(provenance, np.ndarray) or provenance.ndim != 2:
        return {}
    values, counts = np.unique(provenance.astype(np.uint16, copy=False), return_counts=True)
    result: dict[str, int] = {}
    for value, count in zip(values.tolist(), counts.tolist()):
        code = int(value)
        if code == 0:
            label = "ORIGINAL_MAIN"
        elif code == 65534:
            label = "SYMMETRY"
        elif code == 65535:
            label = "MODEL_INFERRED"
        else:
            label = f"ORIGINAL_REFERENCE_{code}"
        result[label] = int(count)
    return result


def _technical_payload(workspace, report) -> dict[str, Any]:
    metadata = workspace.metadata
    return {
        "format": "ConservativeFaceStudio final provenance",
        "version": 1,
        "original_information_confidence": report.as_dict(),
        "source_pixel_counts": _source_counts(workspace),
        "source_legend": {
            "0": "ORIGINAL_MAIN",
            "1..9": "ORIGINAL_REFERENCE_N or RESTORED_REFERENCE backed by an original source",
            "65534": "SYMMETRY",
            "65535": "MODEL_INFERRED",
        },
        "primary_contract": _json_safe(metadata.get("primary_contract")),
        "runtime_source_order": _json_safe(metadata.get("runtime_source_order")),
        "specific_reference_memory": _json_safe(metadata.get("specific_reference_memory")),
        "component_reference_bank": _json_safe(metadata.get("component_reference_bank")),
        "component_alignment_diagnostics": _json_safe(metadata.get("component_alignment_diagnostics")),
        "cross_reference_preclean": _json_safe(metadata.get("cross_reference_preclean")),
        "tiny_observed_evidence": _json_safe(metadata.get("tiny_observed_evidence")),
        "inpaint": {
            "observed_mask": _json_safe(metadata.get("inpaint_observed_mask")),
            "generated_mask": _json_safe(metadata.get("inpaint_generated_mask")),
            "symmetry_mask": _json_safe(metadata.get("inpaint_symmetry_mask")),
            "unresolved_mask": _json_safe(metadata.get("inpaint_unresolved_mask")),
        },
        "policy": "OBSERVE -> RECOVER -> ALIGN -> VALIDATE -> FUSE -> RESTORE -> INFER",
    }


def install_evidence_confidence_runtime() -> None:
    """Attach provenance-based Original Information Confidence to final export.

    The score is not a perceptual realism score. Only the imported MAIN IMAGE and
    reference-backed pixels count as original information. Symmetry, generated and
    unresolved pixels are reported separately. A technical final_provenance.json is
    written beside the final image for audit/debugging and attached as the primary
    provenance sidecar used by the block archive.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from app.strict_execution import StrictBlockExecutor

    original_init = StrictBlockExecutor.__init__

    @wraps(original_init)
    def patched_init(self, workspace, *, history_limit: int = 12) -> None:
        original_init(self, workspace, history_limit=history_limit)
        export_handler = self._handlers.get(BlockKind.EXPORT)
        if export_handler is None:
            return

        @wraps(export_handler)
        def export_with_confidence(block, parameters):
            accounting = reconcile_evidence_accounting(self.workspace)
            report = compute_evidence_confidence(self.workspace)
            self.workspace.metadata["evidence_confidence"] = report.as_dict()
            result = export_handler(block, parameters)
            details = dict(result.details)
            details["evidence_confidence"] = report.as_dict()
            details["evidence_accounting"] = accounting

            output_path = Path(details["path"])
            technical_path = output_path.with_name("final_provenance.json")
            technical_path.write_text(
                json.dumps(_technical_payload(self.workspace, report), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            old_sidecar = details.get("provenance_path")
            if old_sidecar:
                details["project_provenance_path"] = str(old_sidecar)
            details["provenance_path"] = str(technical_path)
            details["technical_provenance_path"] = str(technical_path)

            if self.project.operations:
                self.project.operations[-1].parameters["evidence_confidence"] = report.as_dict()
                self.project.operations[-1].parameters["technical_provenance_path"] = str(technical_path)
            return ExecutionResult(result.block, result.image, details)

        self._handlers[BlockKind.EXPORT] = export_with_confidence

    StrictBlockExecutor.__init__ = patched_init
    _INSTALLED = True
