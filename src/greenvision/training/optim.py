"""AdamW optimizer builder with discriminative LRs and weight-decay exclusion."""

import torch
import torch.nn as nn

WEIGHT_DECAY: float = 1e-4
HEAD_LR: float = 1e-3
BACKBONE_LR: float = 1e-4


def _no_decay_param_names(model: nn.Module) -> set[str]:
    """Return parameter names that should not receive weight decay.

    Normalization layers (BatchNorm/LayerNorm/GroupNorm) carry scale and shift
    parameters that should not be decayed; decaying them harms training.

    Args:
        model: The model to inspect.

    Returns:
        The set of fully-qualified parameter names belonging to normalization
        modules.
    """
    no_decay: set[str] = set()
    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
        nn.GroupNorm,
    )
    for module_name, module in model.named_modules():
        if isinstance(module, norm_types):
            for pname, _ in module.named_parameters(prefix=module_name, recurse=False):
                no_decay.add(pname)
    return no_decay


def build_optimizer(
    model: nn.Module,
    head_lr: float = HEAD_LR,
    backbone_lr: float = BACKBONE_LR,
    wd: float = WEIGHT_DECAY,
) -> torch.optim.AdamW:
    """Build an AdamW optimizer with four discriminative parameter groups.

    The groups are: backbone weights (decayed), backbone biases/norms
    (un-decayed), head weights (decayed), and head biases/norms (un-decayed).
    Backbone groups use ``backbone_lr`` and head groups use ``head_lr``. Only
    parameters with ``requires_grad=True`` are included, so the optimizer
    respects the current freeze state.

    Args:
        model: The model whose parameters are optimized.
        head_lr: Learning rate for the classifier head.
        backbone_lr: Learning rate for the backbone.
        wd: Weight decay applied to the decayed groups.

    Returns:
        A configured ``torch.optim.AdamW`` instance.
    """
    no_decay_names = _no_decay_param_names(model)

    groups: dict[str, list] = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "head_decay": [],
        "head_no_decay": [],
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        in_head = name.startswith("classifier")
        is_no_decay = name.endswith(".bias") or name in no_decay_names
        key = ("head" if in_head else "backbone") + (
            "_no_decay" if is_no_decay else "_decay"
        )
        groups[key].append(p)

    return torch.optim.AdamW(
        [
            {"params": groups["backbone_decay"], "lr": backbone_lr, "weight_decay": wd},
            {
                "params": groups["backbone_no_decay"],
                "lr": backbone_lr,
                "weight_decay": 0.0,
            },
            {"params": groups["head_decay"], "lr": head_lr, "weight_decay": wd},
            {"params": groups["head_no_decay"], "lr": head_lr, "weight_decay": 0.0},
        ]
    )
