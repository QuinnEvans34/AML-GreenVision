"""Reproducibility helpers.

``set_seed`` is called once at the start of every training run so that data
splits, weight initialization, and augmentation sampling are reproducible.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs.

    Args:
        seed: The seed applied to every RNG.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
