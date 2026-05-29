"""Backbone freeze/unfreeze helpers for the two-phase training schedule.

Phase 1 trains only the new head on a fully frozen backbone. Phase 2 gradually
unfreezes the backbone from the top down so the pretrained features adapt
without being wiped out by large early gradients.
"""

import torch.nn as nn


def freeze_all_backbone(model: nn.Module) -> None:
    """Freeze the entire backbone, leaving only the head trainable (Phase 1).

    Every parameter whose name does not start with ``"classifier"`` has its
    gradient disabled, and all backbone ``BatchNorm2d`` layers are switched to
    eval mode so their running statistics stay locked to the pretrained values.

    Args:
        model: The model to freeze in place.
    """
    for name, p in model.named_parameters():
        if not name.startswith("classifier"):
            p.requires_grad = False
    for m in model.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def unfreeze_from_block(model: nn.Module, from_idx: int) -> None:
    """Unfreeze ``features[from_idx:]`` plus the head (Phase 2, gradual).

    Blocks at or after ``from_idx`` become trainable with their BatchNorm layers
    in train mode; earlier blocks stay frozen with BatchNorm in eval mode. The
    head is always trainable.

    Schedule:
        ``from_idx=6`` → Phase 2a (last 2 MBConv stages + final 1x1 conv)
        ``from_idx=3`` → Phase 2b (middle MBConv stages also unfrozen)
        ``from_idx=0`` → Phase 2c (everything unfrozen)

    Args:
        model: The model to modify in place.
        from_idx: Index into ``model.features`` from which to unfreeze.
    """
    for i, block in enumerate(model.features):
        unfreeze = i >= from_idx
        for p in block.parameters():
            p.requires_grad = unfreeze
        for m in block.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train() if unfreeze else m.eval()
    for p in model.classifier.parameters():
        p.requires_grad = True
