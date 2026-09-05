from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def _module():
    sys.path.insert(0, str(RESEARCH))
    try:
        return importlib.import_module("phase04_deeplab_challenger")
    finally:
        if sys.path and sys.path[0] == str(RESEARCH):
            sys.path.pop(0)


def test_official_deeplab_builder_executes_18_class_cpu_forward() -> None:
    module = _module()
    import torch

    assert module.CLASS_COUNT == 18
    network = module._new_network().eval()
    with torch.inference_mode():
        output = network(torch.zeros((1, 3, 64, 64), dtype=torch.float32))["out"]
    assert tuple(output.shape) == (1, 18, 64, 64)
    assert module.parameter_count(network) > 5_000_000


def test_adapter_is_offline_official_upstream_and_fail_closed() -> None:
    module = _module()
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert module.ARCHITECTURE == "torchvision_deeplabv3_mobilenet_v3_large"
    assert "deeplabv3_mobilenet_v3_large" in source
    assert "weights=None" in source
    assert "weights_backbone=None" in source
    assert "network.backbone.load_state_dict(features, strict=True)" in source
    assert "PHASE04_TRAINING_CLASSES" in source
    assert "from_trained_checkpoint" in source
    assert "recorded_classes != list(PHASE04_TRAINING_CLASSES)" in source
