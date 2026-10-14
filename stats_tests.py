"""
stats_tests.py — Statistical tests on bit streams.

Each test returns a p-value; a stream passes at alpha = 0.01 (the NIST
SP 800-22 convention). Over K independent trials, NIST's acceptable pass
proportion is  p_hat ± 3*sqrt(p_hat(1-p_hat)/K)  with p_hat = 0.99,
i.e. at least 19/20 passes for K = 20.

monobit_chi2  : chi-square goodness of fit for P(0) = P(1). Detects bias.
runs          : Wald–Wolfowitz runs test. Detects too few/many runs.
autocorr      : lag-1 autocorrelation z-test. Detects bit-to-bit dependence.
serial        : NIST SP 800-22 §2.11 serial test (m = 2), first statistic.
                Detects non-uniform 2-bit pattern frequencies.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaincc
from scipy.stats import chisquare, norm

ALPHA = 0.01


def monobit_chi2(bits) -> float:
    n = len(bits); n1 = int(np.sum(bits))
    return float(chisquare([n - n1, n1], [n / 2, n / 2]).pvalue)


def runs(bits) -> float:
    b = np.asarray(bits, dtype=np.int8)
    n = len(b); n1 = int(b.sum()); n0 = n - n1
    if n0 == 0 or n1 == 0:
        return 0.0
    R = 1 + int(np.sum(b[1:] != b[:-1]))
    mu = 2 * n0 * n1 / n + 1
    var = 2 * n0 * n1 * (2 * n0 * n1 - n) / (n**2 * (n - 1))
    return float(2 * norm.sf(abs((R - mu) / np.sqrt(var))))


def lag1_autocorr(bits) -> float:
    x = np.asarray(bits, dtype=float) * 2 - 1
    x = x - x.mean()
    denom = np.dot(x, x)
    return float(np.dot(x[:-1], x[1:]) / denom) if denom else 1.0


def autocorr(bits) -> float:
    r = lag1_autocorr(bits)
    return float(2 * norm.sf(abs(r) * np.sqrt(len(bits))))


def _psi2(b, m):
    if m == 0:
        return 0.0
    n = len(b)
    ext = np.concatenate([b, b[: m - 1]])
    idx = np.zeros(n, dtype=np.int64)
    for k in range(m):
        idx = (idx << 1) | ext[k: k + n]
    counts = np.bincount(idx, minlength=2**m)
    return (2**m / n) * np.sum(counts.astype(float) ** 2) - n


def serial(bits, m: int = 2) -> float:
    b = np.asarray(bits, dtype=np.int64)
    d1 = _psi2(b, m) - _psi2(b, m - 1)
    return float(gammaincc(2 ** (m - 2), d1 / 2))


TESTS = {"Monobit χ²": monobit_chi2, "Runs": runs,
         "Autocorr": autocorr, "Serial": serial}


def run_all(bits) -> dict[str, float]:
    return {name: f(bits) for name, f in TESTS.items()}
