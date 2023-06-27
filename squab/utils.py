"""Utility functions.
"""

from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def mixup(
    x: Tensor,
    y: Tensor,
    alpha: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, float]:
    """Apply mixup data augmentation, constructing 'virtual examples'.

    Args:
        x: Inputs.
        y: Targets.
        alpha: Strength of interpolation between feature-target pairs in the range
            [0, inf].

    Returns:
        x_virtual: Linear interpolation of feature vectors.
        y: Original targets.
        y_virtual: Target of interpolated features.
        lambda: Interpolation factor in the range [0, 1]; used by `mixup_loss`.


    Reference:
        Zhang, Hongyi et al. "mixup: Beyond Empirical Risk Minimization". 2018.
    """
    if alpha < 0:
        raise ValueError("alpha must be >=0")
    l = 1
    if alpha > 0:
        l = np.random.beta(alpha, alpha)

    # permute indices and interpolate features
    indices = torch.randperm(x.size(0))
    x_virtual = l * x + (1 - l) * x[indices, :]

    return x_virtual, y, y[indices], l


def mixup_loss(crit, pred: Tensor, y: Tensor, y_virtual: Tensor, l: float) -> Tensor:
    """Apply loss function `crit` given predictions for the two sets of mixup targets.

    Args:
        crit: Loss function.
        pred: Predicted targets.
        ya: Original targets.
        yb: Target of interpolated features.
        l: Interpolation factor.

    Returns:
        Loss with respect to ya, yb scaled by `l`
    """
    return l * crit(pred, y) + (1 - l) * crit(pred, y_virtual)
