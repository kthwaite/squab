"""
"""

import hashlib
from typing import Any, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import Linear, Module


class TritProjector:
    """Project tokenized strings to vectors of ints ∈ {-1, 0, 1}."""

    def __init__(self, hasher: Optional[Any] = None):
        self.hasher = hasher or hashlib.sha256
        # TODO: faster, pussycat! kill! kill!
        self._vmap_project = np.vectorize(self.project, signature="()->(n)")
        self._d = {}

    @property
    def feature_dim(self) -> int:
        # FIXME: this should reflect the `self.hasher` hash size
        return 128

    def project(self, token: str) -> np.ndarray:
        """Transform semi-nibbles from a hashed `token` into a vector of trits.

        Caches strings for speed.
        """
        try:
            return self._d[token]
        except KeyError:
            digest = self.hasher(token.encode()).digest()
            v = np.zeros(128, dtype=np.float32)
            for ix, byte in zip(range(0, 128, 4), digest):
                v[ix] = [0, 1, -1, 0][byte & 3]
                v[ix + 1] = [0, 1, -1, 0][(byte >> 2) & 3]
                v[ix + 2] = [0, 1, -1, 0][(byte >> 4) & 3]
                v[ix + 3] = [0, 1, -1, 0][(byte >> 6) & 3]
            self._d[token] = v
            return v

    def __call__(self, tokens: List[List[str]]) -> Tensor:
        return torch.from_numpy(self._vmap_project(tokens))


class TritProjectionEmbedding(Module):
    """Trit-projection -> embedding."""

    # FIXME: make hasher customizable here
    def __init__(self, trit_dim: int, embedding_dim: int):
        """
        Args:
            trit_dim: Dimension of trit vectors.
            embedding_dim: Embedding dimension derived from trits.
        """
        super().__init__()
        self.projector = TritProjector()
        self.embedding = Linear(trit_dim, embedding_dim)

    def forward(self, x: List[List[str]]) -> Tensor:
        p = self.projector(x)
        return self.embedding(p)
