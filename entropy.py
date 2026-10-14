"""
entropy.py — Entropy estimators.

shannon          : H = -sum p log2 p over single bits. Blind to correlation.
mcv_min_entropy  : NIST SP 800-90B §6.3.1 Most Common Value estimate, using
                   the 99% upper confidence bound on the most likely symbol.
markov_min_entropy: NIST SP 800-90B §6.3.3 Markov estimate (binary case):
                   fits a first-order Markov chain and finds the most likely
                   128-bit sequence. Catches bit-to-bit correlation that the
                   MCV estimate misses.
conservative     : min of the two — what we feed the Toeplitz extractor.
"""

from __future__ import annotations

import numpy as np


def shannon(bits: np.ndarray) -> float:
    p1 = float(np.mean(bits))
    if p1 in (0.0, 1.0):
        return 0.0
    return float(-p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1))


def mcv_min_entropy(bits: np.ndarray) -> float:
    n = len(bits)
    p_hat = max(np.mean(bits), 1 - np.mean(bits))
    p_u = min(1.0, p_hat + 2.576 * np.sqrt(p_hat * (1 - p_hat) / (n - 1)))
    return float(-np.log2(p_u))


def markov_min_entropy(bits: np.ndarray, L: int = 128) -> float:
    b = np.asarray(bits, dtype=np.int64)
    P = [np.mean(b == 0), np.mean(b == 1)]
    a, c = b[:-1], b[1:]
    T = np.zeros((2, 2))
    for i in (0, 1):
        o_i = np.sum(a == i)
        if o_i == 0:
            T[i] = [1.0, 1.0]
            continue
        for j in (0, 1):
            T[i, j] = np.sum((a == i) & (c == j)) / o_i
    # max-probability path of length L, in log space
    logT = np.log2(np.clip(T, 1e-300, 1))
    v = np.log2(np.clip(P, 1e-300, 1))
    for _ in range(L - 1):
        v = np.max(v[:, None] + logT, axis=0)
    p_max = np.max(v)
    return float(min(-p_max / L, 1.0))


def conservative(bits: np.ndarray) -> float:
    return min(mcv_min_entropy(bits), markov_min_entropy(bits))
