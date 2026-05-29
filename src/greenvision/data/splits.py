"""Stratified train/val/test splitting for ImageFolder datasets."""

from torchvision.datasets import ImageFolder

from sklearn.model_selection import train_test_split


def stratified_split(
    dataset: ImageFolder,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Stratified train/val/test split of an ImageFolder by class label.

    The split is performed in two stratified passes: first the test set is
    peeled off the full index, then the validation set is peeled off the
    remaining train+val indices. Both passes preserve per-class proportions.

    Args:
        dataset: An ``ImageFolder`` whose ``samples`` attribute provides the
            ``(path, label)`` pairs used for stratification.
        val_frac: Fraction of the *full* dataset to allocate to validation.
        test_frac: Fraction of the *full* dataset to allocate to test.
        seed: Random seed forwarded to scikit-learn for reproducibility.

    Returns:
        A ``(train_idx, val_idx, test_idx)`` tuple of index lists that can wrap
        the dataset in ``torch.utils.data.Subset``.
    """
    labels = [label for _, label in dataset.samples]
    indices = list(range(len(dataset)))

    # First: peel off test, stratified on the full label set.
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_frac,
        stratify=labels,
        random_state=seed,
    )

    # Then: peel off val from the remainder, stratified on the remainder labels.
    tv_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_frac / (1 - test_frac),
        stratify=tv_labels,
        random_state=seed,
    )
    return train_idx, val_idx, test_idx
