"""
"""

from torch import Tensor
from torch.nn import Module, Sequential


class Residual(Module):
    def __init__(self, mod: Module):
        super().__init__()
        self.mod = mod

    def forward(self, x: Tensor) -> Tensor:
        return self.mod(x) + x


class SequentialResidual(Residual):
    def __init__(self, *mod: Module):
        super().__init__(Sequential(*mod))
