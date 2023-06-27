"""Linear layers and helpers.
"""

from typing import Type

from torch.nn import GELU, Dropout, LayerNorm, Linear, Module, ReLU

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
