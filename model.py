"""Model factory for semantic segmentation backbones via segmentation_models_pytorch."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


SUPPORTED_ARCHS = {"deeplabv3plus", "unet"}
SUPPORTED_ENCODERS = {"resnet34", "efficientnet-b0", "efficientnet-b3"}


def build_model(
    architecture: str,
    encoder_name: str,
    num_classes: int,
    in_channels: int = 3,
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Build a segmentation model.

    Args:
        architecture: "deeplabv3plus" or "unet".
        encoder_name: e.g. "resnet34" or "efficientnet-b0".
        num_classes: Number of semantic classes.
        in_channels: Input channels.
        encoder_weights: Pretrained encoder weights.
    """
    arch = architecture.lower()
    enc = encoder_name.lower()

    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unsupported architecture '{architecture}'. Supported: {sorted(SUPPORTED_ARCHS)}")
    if enc not in SUPPORTED_ENCODERS:
        raise ValueError(f"Unsupported encoder '{encoder_name}'. Supported: {sorted(SUPPORTED_ENCODERS)}")

    common_kwargs = dict(
        encoder_name=enc,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )

    if arch == "deeplabv3plus":
        model = smp.DeepLabV3Plus(**common_kwargs)
    else:
        model = smp.Unet(**common_kwargs)

    return model
