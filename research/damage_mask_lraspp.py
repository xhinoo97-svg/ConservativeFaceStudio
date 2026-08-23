from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from damage_mask_lraspp_contract import (
    BACKBONE_SHA256,
    TORCH_VERSION,
    TORCHVISION_VERSION,
    verify_file,
    verify_backbone_checkpoint,
)


def _new_network(*, classes: int) -> nn.Module:
    import torchvision
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    if str(torch.__version__) != TORCH_VERSION:
        raise RuntimeError(f"Unexpected torch version: {torch.__version__} != {TORCH_VERSION}")
    if str(torchvision.__version__) != TORCHVISION_VERSION:
        raise RuntimeError(
            f"Unexpected torchvision version: {torchvision.__version__} != {TORCHVISION_VERSION}"
        )
    return lraspp_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=int(classes),
    )


class LRASPPDamageModel(nn.Module):
    """Thin CFS adapter around torchvision's official LR-ASPP implementation.

    The public input contract remains an unnormalized RGB float tensor in [0, 1],
    matching the existing DamageMask development dataset and ONNX runtime adapter.
    """

    def __init__(self, *, classes: int, backbone_checkpoint: Path) -> None:
        super().__init__()
        self.backbone_sha256 = verify_backbone_checkpoint(Path(backbone_checkpoint))
        if self.backbone_sha256 != BACKBONE_SHA256:
            raise RuntimeError("Backbone checkpoint verification did not return the frozen digest")

        network = _new_network(classes=int(classes))
        payload = torch.load(str(backbone_checkpoint), map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError("Official MobileNetV3 checkpoint is not a state dictionary")
        features = {
            key[len("features.") :]: value
            for key, value in payload.items()
            if str(key).startswith("features.")
        }
        if not features:
            raise RuntimeError("Official MobileNetV3 checkpoint contains no feature tensors")
        network.backbone.load_state_dict(features, strict=True)
        self.network = network
        self.loaded_backbone_tensor_count = len(features)
        self._register_normalization()

    def _register_normalization(self) -> None:
        self.register_buffer(
            "normalization_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "normalization_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )

    @classmethod
    def from_trained_checkpoint(
        cls,
        *,
        checkpoint: Path,
        expected_sha256: str,
        classes: int,
    ) -> "LRASPPDamageModel":
        verify_file(Path(checkpoint), expected_sha256=expected_sha256)
        payload = torch.load(str(checkpoint), map_location="cpu")
        if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
            raise RuntimeError("Trained LR-ASPP checkpoint has no state_dict")
        recorded_classes = payload.get("classes")
        if not isinstance(recorded_classes, list) or len(recorded_classes) != int(classes):
            raise RuntimeError("Trained LR-ASPP checkpoint taxonomy does not match runtime")

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.network = _new_network(classes=int(classes))
        instance.loaded_backbone_tensor_count = 0
        instance.backbone_sha256 = str(payload.get("backbone_sha256", ""))
        instance._register_normalization()
        instance.load_state_dict(payload["state_dict"], strict=True)
        instance.eval()
        return instance

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        normalized = (image - self.normalization_mean) / self.normalization_std
        return self.network(normalized)["out"]


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
