"""EfficientNet-B0 backbone with a PlantVillage-sized classification head."""

import torch.nn as nn

from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

NUM_CLASSES: int = 39  # 38 PlantVillage classes + 1 Background_without_leaves (negative) class
FEATURE_DIM: int = 1280  # EfficientNet-B0 feature dimension feeding the classifier


def build_model(num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> nn.Module:
    """Build EfficientNet-B0 with a fresh head sized for our class count.

    The ImageNet-pretrained backbone is loaded unchanged; only the classifier
    is replaced with a ``Dropout -> Linear`` head producing ``num_classes``
    logits. The backbone remains accessible as ``model.features`` and the head
    as ``model.classifier``, which the freeze/unfreeze helpers rely on.

    Args:
        num_classes: Number of output logits (defaults to ``NUM_CLASSES``).
        dropout: Dropout probability applied before the final linear layer.

    Returns:
        An ``nn.Module`` whose ``forward`` maps ``[B, 3, 224, 224]`` to
        ``[B, num_classes]``.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(FEATURE_DIM, num_classes),
    )
    return model
