from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "research-refface-cpu-vertical-slice.yml"
SCRIPT = ROOT / "research" / "run_refface_cpu_vertical_slice.py"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_refface_workflow_is_manual_only_until_damage_mask_gate_is_verified() -> None:
    payload = yaml.safe_load(_workflow_text())
    triggers = payload.get("on") or payload.get(True)
    assert isinstance(triggers, dict)
    assert set(triggers) == {"workflow_dispatch"}


def test_refface_source_and_cpu_patch_are_exactly_pinned() -> None:
    text = _workflow_text()
    assert "0f1ad75677cc8fae4ae14d878e4c6cfce9365f28" in text
    assert "torch.cuda.FloatTensor(bs, self.n_classes, h, w).zero_()" in text
    assert "x.new_zeros((bs, self.n_classes, h, w))" in text
    assert "Expected exactly one CUDA one-hot allocation" in text
    assert "grep -R --line-number -E 'torch\\.cuda\\.FloatTensor|\\.cuda\\('" in text


def test_refface_uses_verified_cfs_identity_and_parsing_assets() -> None:
    text = _workflow_text()
    assert "opencv_yunet" in text
    assert "opencv_sface" in text
    assert "face_parsing_resnet18_onnx" in text
    assert "checksum_ok" in text
    assert "face_parsing_resnet18.onnx" in text


def test_refface_uses_two_same_identity_controlface_views_not_final_holdout() -> None:
    text = _workflow_text()
    assert "CONTROLFACE_REVISION" in text
    assert "No multi-view female ControlFace identity found" in text
    assert "len(values) >= 2" in text
    assert "final_holdout_used" in text
    script = SCRIPT.read_text(encoding="utf-8")
    assert '"final_holdout_used": False' in script
    assert "same_identity_reference_gate" in script


def test_refface_enforces_frozen_identity_and_total_pc_resource_contract() -> None:
    text = _workflow_text()
    script = SCRIPT.read_text(encoding="utf-8")
    assert "detect_resource_budget(0.80)" in script
    assert "assert_memory_within_budget" in script
    assert "FACE_MODEL_DEFAULTS.sface_same_identity_cosine" in script
    assert "healthy_outside_exact_main" in script
    assert "GENERATED_MODEL_INFERRED" in script
    assert "r['identity']['threshold'] == 0.363" in text
    assert "r['runtime']['resource_budget']['max_fraction'] == 0.80" in text


def test_refface_only_loads_unetg_and_arcface_for_inference() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    assert "from networks.UnetG import UnetG" in script
    assert "from networks.arcface_models import resnet101" in script
    assert "Discriminator" not in script
    assert "netD" not in script
