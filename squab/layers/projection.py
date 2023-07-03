"""
"""

import hashlib
from typing import Any, List, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Linear, Module

from .skipgram_convolution import SkipgramConvolution


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

    def __call__(self, tokens: List[List[str]]) -> np.ndarray:
        return self._vmap_project(tokens)


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
        self.embedding = Linear(trit_dim, embedding_dim, bias=False)

    def forward(self, x: List[List[str]]) -> Tensor:
        p = torch.from_numpy(self.projector(x)).to("cuda")
        return self.embedding(p)


class ProjAttention(Module):
    def __init__(self, in_channels, out_channels, embedding_dim, mask):
        super().__init__()
        self.feat = SkipgramConvolution(in_channels, out_channels, embedding_dim, mask)
        self.attn = SkipgramConvolution(in_channels, out_channels, embedding_dim, mask)
        self.act = nn.Softmax(dim=-1)

    def forward(self, x):
        feat = self.feat(x).squeeze(dim=-1)
        attn = self.act(self.attn(x).squeeze(dim=-1))
        return torch.sum(attn * feat, dim=-1)


PRIM = np.int64((1 << 61) - 1)
CEIL = np.int64((1 << 32) - 1)


class MinHash:
    """MinHash implemented as Hash + LCG."""

    def __init__(
        self,
        dim: int = 128,
        seed: int = 0xDEADBEEF,
        hash_fn: Callable[[str], int] = farmhash.hash32,
    ):
        self.seed = seed
        self.dim = dim
        self.hash_fn = hash_fn
        self.lcg_mul, self.lcg_inc = self.init_lcg(dim, seed)
        self._state = np.full(dim, CEIL, dtype=np.int64)

    def clear(self):
        self._state.fill(CEIL)

    def digest(self):
        return np.copy(self._state)

    def state(self):
        return self._state

    @classmethod
    def init_lcg(cls, dim: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create the multiplier and increment values for a linear congruential
        generator.

        Returns:
            Arrays for multiplier and increment values respectively.
        """
        gen = np.random.RandomState(seed)
        mul = np.zeros(dim, dtype=np.int64)
        inc = np.zeros(dim, dtype=np.int64)
        for index in range(dim):
            mul[index] = gen.randint(1, PRIM, dtype=np.int64)
            inc[index] = gen.randint(0, PRIM, dtype=np.int64)
        return mul, inc

    def update(self, s: str):
        value = self.hash_fn(s)
        value = np.bitwise_and((self.lcg_mul * value + self.lcg_inc) & PRIM, CEIL)
        np.minimum(value, self._state, out=self._state)


class TokenFingerprinter:
    """Token 'fingerprinting' as described in the pNLP-Mixer paper."""

    def __init__(
        self,
        feature_dim: int = 256,
        ngrams: int = 3,
        dim: int = 128,
        **kwargs,
    ):
        """
        Args:
            feature_dim: Dimension of fingerprint vectors. Defaults to 256.
            ngrams: N-gram size to use when splitting subwords. Defaults to 3.
            dim: Dimension of MinHash vectors. Defaults to 128.
            kwargs: Passed to MinHash constructor.
        """
        if dim % 2 != 0:
            raise ValueError(f"Dim must be a power of 2")
        self.feature_dim = feature_dim
        self.ngrams = ngrams
        self.hash = MinHash(dim, **kwargs)
        self._filter = np.eye(feature_dim, dtype=np.float32)
        self._ngrams = np.zeros((32, self.hash.dim), dtype=np.int64)

    def filter(self, value: np.ndarray):
        return self._filter[value & (self.feature_dim - 1)].sum(axis=-2)

    def __call__(self, token: str, is_continuation: bool = False) -> np.ndarray:
        """Fingerprint a token.

        Continuations are hashed 'as-is', whereas other tokens are split into n-grams
        with the fingerprint set to the minimum for each element over all n-gram hashes.

        Args:
            token: Subword produced by tokenizer.
            is_continuation: Flag if this is a continuation token, e.g. '##ing'.
        """
        if is_continuation or len(token) < self.ngrams:
            self.hash.update(token)
            value = self._filter[self.hash.state() & (self.feature_dim - 1)].sum(
                axis=-2
            )
            self.hash.clear()
            return value
        tok_len = len(token) - self.ngrams + 1
        for index in range(tok_len):
            self.hash.update(token[index : index + self.ngrams])
            self._ngrams[index, :] = self.hash.state()
            self.hash.clear()
        return self._filter[
            self._ngrams[:tok_len].min(axis=-2) & (self.feature_dim - 1)
        ].sum(axis=-2)
