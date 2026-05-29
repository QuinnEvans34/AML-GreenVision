"""Phase 1 verification: confirm the data layer yields correct batches.

Run from the repo root once data/raw/PlantVillage/ is populated:
    .venv/bin/python scripts/_phase1_verify.py

NOTE: the body is wrapped in `if __name__ == "__main__":` because macOS
defaults to the "spawn" multiprocessing start method, and DataLoader workers
re-import this script on spawn. Without the guard, each worker would try to
build its own DataLoader → spawn its own workers → BrokenPipeError.
"""

from greenvision.data.datasets import build_dataloaders


def main() -> None:
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        "data/raw/PlantVillage", batch_size=64
    )
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (64, 3, 224, 224), batch_x.shape
    assert batch_x.dtype.is_floating_point, batch_x.dtype
    print(f"Classes: {len(class_names)} (expect 39)")
    print(
        f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
    )
    print(f"Per-channel mean: {batch_x.mean(dim=[0, 2, 3]).tolist()}")  # near 0
    print(f"Per-channel std:  {batch_x.std(dim=[0, 2, 3]).tolist()}")  # near 1


if __name__ == "__main__":
    main()
