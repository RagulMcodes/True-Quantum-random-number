"""
extractors.py — Classical post-processing (randomness extraction).

von_neumann  : Von Neumann (1951). Unbiased output *if* input bits are IID.
peres        : Peres (1992) iterated VN. Same IID assumption, but recycles
               discarded information so efficiency approaches the Shannon
               entropy of the source instead of p(1-p).
toeplitz     : Toeplitz-hashing extractor (a 2-universal hash family).
               By the Leftover Hash Lemma, if each n-bit input block has
               min-entropy >= k, the m = k - 2*log2(1/eps) output bits are
               eps-close to uniform — with NO independence assumption on the
               input bits. This is the extractor used in commercial and
               research QRNGs (e.g. Ma et al., npj Quantum Inf. 2016).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz as _toeplitz


def von_neumann(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8)
    b = b[: len(b) // 2 * 2].reshape(-1, 2)
    keep = b[:, 0] != b[:, 1]
    return b[keep, 0].copy()          # 01 -> 0, 10 -> 1


def peres(bits: np.ndarray, depth: int = 8) -> np.ndarray:
    """Peres' iterated von Neumann procedure (Ann. Statist. 20(1), 1992)."""
    b = np.asarray(bits, dtype=np.uint8)
    if depth == 0 or len(b) < 2:
        return np.empty(0, dtype=np.uint8)
    pairs = b[: len(b) // 2 * 2].reshape(-1, 2)
    differ = pairs[:, 0] != pairs[:, 1]
    vn_out = pairs[differ, 0]
    xor_seq = pairs[:, 0] ^ pairs[:, 1]           # "did the pair differ?"
    same_seq = pairs[~differ, 0]                  # value of equal pairs
    return np.concatenate([vn_out,
                           peres(xor_seq, depth - 1),
                           peres(same_seq, depth - 1)])


class ToeplitzExtractor:
    """
    Seeded strong extractor. The seed (n + m - 1 bits) must be uniform and
    independent of the source, but it may be public and reused across blocks
    (strong-extractor property). We draw it once from a fixed-seed generator
    so results are reproducible.
    """

    def __init__(self, n: int = 1024, security_bits: int = 32, seed: int = 2026):
        self.n = n
        self.security_bits = security_bits    # eps = 2^-security_bits
        self._seed_rng = np.random.default_rng(seed)
        self._cache: dict[int, np.ndarray] = {}

    def output_len(self, h_min_per_bit: float) -> int:
        k = np.floor(self.n * h_min_per_bit)
        return int(max(0, k - 2 * self.security_bits))

    def _matrix(self, m: int) -> np.ndarray:
        if m not in self._cache:
            s = self._seed_rng.integers(0, 2, self.n + m - 1, dtype=np.uint8)
            col, row = s[:m], s[m - 1:]
            self._cache[m] = _toeplitz(col, row).astype(np.int32)   # m x n
        return self._cache[m]

    def extract(self, bits: np.ndarray, h_min_per_bit: float) -> np.ndarray:
        m = self.output_len(h_min_per_bit)
        nb = len(bits) // self.n
        if m == 0 or nb == 0:
            return np.empty(0, dtype=np.uint8)
        X = np.asarray(bits[: nb * self.n], dtype=np.int32).reshape(nb, self.n)
        Y = (X @ self._matrix(m).T) & 1
        return Y.astype(np.uint8).ravel()
