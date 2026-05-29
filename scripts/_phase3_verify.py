"""Phase 3 verification: confirm freeze/unfreeze + optimizer grouping work.

Run from the repo root:
    .venv/bin/python scripts/_phase3_verify.py
"""

from greenvision.models.efficientnet_head import build_model
from greenvision.training.optim import build_optimizer
from greenvision.training.phases import freeze_all_backbone, unfreeze_from_block

model = build_model()
# Phase 1 freeze
freeze_all_backbone(model)
trainable = sum(1 for p in model.parameters() if p.requires_grad)
all_p = sum(1 for _ in model.parameters())
print(f"Phase 1 — trainable: {trainable}/{all_p} (expect head-only, small number)")
opt = build_optimizer(model)
print(
    f"Phase 1 — optimizer param groups: {[len(g['params']) for g in opt.param_groups]}"
)

# Phase 2a unfreeze
unfreeze_from_block(model, from_idx=6)
trainable_2a = sum(1 for p in model.parameters() if p.requires_grad)
print(f"Phase 2a — trainable: {trainable_2a}/{all_p} (expect more than phase 1)")
