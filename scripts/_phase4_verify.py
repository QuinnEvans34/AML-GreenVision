"""Phase 4 verification: 1-epoch smoke test on a tiny subset.

Run from the repo root once data/raw/PlantVillage/ is the 39-class layout:
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/_phase4_verify.py

NOTE: body is wrapped in `if __name__ == "__main__":` so the macOS spawn
multiprocessing start method doesn't re-import this and recursively spawn
DataLoader workers (BrokenPipeError otherwise).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from greenvision.data.datasets import build_dataloaders
from greenvision.models.efficientnet_head import build_model
from greenvision.training.loop import evaluate, train_one_epoch


def main() -> None:
    train_loader, val_loader, _, _ = build_dataloaders(
        "data/raw/PlantVillage", batch_size=32
    )
    # Use a tiny subset for the smoke test
    train_loader = DataLoader(
        Subset(train_loader.dataset, range(200)), batch_size=32, shuffle=True
    )
    val_loader = DataLoader(Subset(val_loader.dataset, range(50)), batch_size=32)

    device = torch.device("mps")
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    m_train = train_one_epoch(model, train_loader, opt, crit, device)
    m_val = evaluate(model, val_loader, crit, device)
    print(
        f"Smoke test: train_loss={m_train['train_loss']:.3f}, val_acc={m_val['val_acc']:.3f}"
    )


if __name__ == "__main__":
    main()
