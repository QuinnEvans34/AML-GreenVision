"""Wiring that turns a PlantVillage ImageFolder into train/val/test DataLoaders."""

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from greenvision.data.class_names import save_class_names
from greenvision.data.splits import stratified_split
from greenvision.data.transforms import eval_tfms, train_tfms
from greenvision.training.seed import set_seed


def build_dataloaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build stratified train/val/test DataLoaders from an ImageFolder root.

    Two ``ImageFolder`` views over the same directory are created so that the
    training split receives augmenting ``train_tfms`` while the validation and
    test splits receive deterministic ``eval_tfms``. Both views share the same
    alphabetical class ordering, so the stratified indices are interchangeable.
    The class list is persisted to the canonical path before returning.

    Args:
        root: Path to the ImageFolder root (one subfolder per class).
        batch_size: Batch size for all three loaders.
        num_workers: Worker processes per loader. ``persistent_workers`` is
            enabled only when this is greater than zero.
        seed: Seed applied via ``set_seed`` and forwarded to the split.

    Returns:
        A ``(train_loader, val_loader, test_loader, class_names)`` tuple, where
        ``class_names`` is the alphabetical class list.
    """
    set_seed(seed)

    train_ds = ImageFolder(root, transform=train_tfms)
    eval_ds = ImageFolder(root, transform=eval_tfms)

    train_idx, val_idx, test_idx = stratified_split(train_ds, seed=seed)

    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(eval_ds, val_idx)
    test_subset = Subset(eval_ds, test_idx)

    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )

    class_names = train_ds.classes
    save_class_names(class_names)

    return train_loader, val_loader, test_loader, class_names
