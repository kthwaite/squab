"""Linear layers and helpers.
"""

from typing import Type

import numpy as np
from torch.nn import (GELU, GLU, Dropout, LayerNorm, Linear, Module, ReLU,
                      Sequential)
from torch.nn.utils import weight_norm

from .residual import SequentialResidual


def InverseBottleneck(
    in_features: int,
    expansion_factor: int = 4,
    act: Type[Module] = GELU,
    dropout: float = 0.0,
    layer_norm: bool = True,
    depth: int = 1,
) -> SequentialResidual:
    """Variable-depth 'inverse bottleneck' with residual connection.

    Expands input features by `expansion_factor`, optionally applying a number of
    hidden layers when `depth` is greater than 1.

    Args:
        in_features
        expansion_factor
        act: Activation function. Defaults to GELU.
        dropout: Dropout between layers. Defaults to 0.0.
        layer_norm: Apply layer norm on input. Defaults to True.
        depth: Number of hidden-hidden connections.
    """
    hidden_size = in_features * expansion_factor
    layers = [
        Linear(in_features, hidden_size),
        act(),
        Dropout(dropout),
    ]
    for _ in range(depth - 1):
        layers.extend(
            [
                Linear(hidden_size, hidden_size),
                act(),
                Dropout(dropout),
            ]
        )
    layers.extend(
        [
            Linear(hidden_size, in_features),
            Dropout(dropout),
        ]
    )
    if layer_norm:
        layers = [LayerNorm(in_features)] + layers
    return SequentialResidual(*layers)


def WNLinear(
    in_features: int,
    out_features: int,
    dropout: float = 0.0,
    bias: bool = True,
) -> Linear:
    """Weight-normalized Linear layer."""
    layer = Linear(in_features, out_features, bias=bias)
    layer.weight.data.normal_(mean=0, std=np.sqrt((1 - dropout) / in_features))
    layer.bias.data.zero_()
    return weight_norm(layer)


class GatedLinearStack(Module):
    """Gated weight-normalized layer stack, consisting of three weight-normalized Linear
    layers connected by Gated Linear Units."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.layers = Sequential(
            WNLinear(in_features, out_features * 4, dropout=dropout, bias=bias),
            GLU(),
            WNLinear(out_features * 2, out_features * 2, dropout=dropout, bias=bias),
            GLU(),
            WNLinear(out_features, out_features, dropout=dropout, bias=bias),
        )

    def forward(self, x):
        return self.layers(x)
