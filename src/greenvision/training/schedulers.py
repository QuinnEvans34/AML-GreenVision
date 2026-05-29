"""Learning-rate scheduler builder."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_scheduler(optimizer: Optimizer) -> ReduceLROnPlateau:
    """Build a plateau scheduler that halves the LR on stalled validation loss.

    Call ``scheduler.step(val_loss)`` after each epoch's validation pass. The
    LR is halved when val loss has not improved for two consecutive epochs.

    Args:
        optimizer: The optimizer whose learning rates are adjusted.

    Returns:
        A ``ReduceLROnPlateau`` configured with ``mode="min"``, ``factor=0.5``,
        ``patience=2``.
    """
    return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
