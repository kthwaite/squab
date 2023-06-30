"""
"""
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Conv2d, Module


def create_mask(spec: Tuple[int], dim: int) -> Tensor:
    """Create a tensor mask for a convolutional layer."""
    return (
        torch.tensor(spec, dtype=torch.float, requires_grad=False)
        .reshape(-1, 1)
        .repeat((1, dim))
    )


class SkipgramConvolution(Module):
    def __init__(self, in_channels, out_channels, embedding_dim, mask):
        super().__init__()
        self.mask = create_mask(mask, embedding_dim)
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.mask.shape,
            stride=1,
            padding=(self.mask.size(0) // 2, 0),
        )
        with torch.no_grad():
            self.conv.weight *= self.mask

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
