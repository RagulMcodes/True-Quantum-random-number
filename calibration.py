"""
calibration.py — Our correction method: Closed-Loop Calibration + Extraction.

Stage 1 — fix the source (quantum side)
---------------------------------------
Born rule for Ry(theta)|0>:  P(1) = sin^2(theta / 2)
=> the effective rotation can be read off the measured bias:
       theta_hat = 2 * arcsin( sqrt(p_hat) )
The over-rotation error is  e_hat = theta_hat - pi/2.

We generate bits in blocks. After each block we estimate e_hat from that
block's own output bits and update the commanded angle for the next block:
       c_{k+1} = c_k - g * e_hat_k          (integral feedback, gain g)
This needs NO extra calibration shots: the random bits themselves are the
calibration signal. Because it runs continuously, it also tracks drift.

Stage 2 — clean the residue (classical side)
--------------------------------------------
Calibration cannot remove readout correlation, and the block-level estimate
is noisy. So the calibrated stream goes through a Toeplitz extractor whose
output length is set by a *conservative* min-entropy estimate
(min of NIST MCV and Markov estimates), which accounts for correlation.

Why the two stages help each other: calibration pushes the raw min-entropy
towards 1 bit/bit, so the extractor is allowed to output many more bits than
it could from the uncorrected stream.
"""

from __future__ import annotations

import numpy as np

from device import HADAMARD_ANGLE, FaultyQRNGDevice
from entropy import conservative
from extractors import ToeplitzExtractor


def estimate_angle_error(bits: np.ndarray, commanded: float) -> float:
    p = np.clip(np.mean(bits), 1e-6, 1 - 1e-6)
    theta_hat = 2 * np.arcsin(np.sqrt(p))
    return theta_hat - HADAMARD_ANGLE


def closed_loop_generate(device: FaultyQRNGDevice, n_bits: int,
                         block: int = 500, gain: float = 0.6):
    """Generate n_bits with continuous self-calibration.

    Returns (bits, commanded_angle_log) — the log shows the controller
    tracking the hidden drift.
    """
    c = HADAMARD_ANGLE
    out, log = [], []
    for start in range(0, n_bits, block):
        k = min(block, n_bits - start)
        bits = device.run(c, k)
        out.append(bits)
        log.append((device.t, c))
        c -= gain * estimate_angle_error(bits, c)
    return np.concatenate(out), log


def our_method(device: FaultyQRNGDevice, n_bits: int,
               extractor: ToeplitzExtractor, block: int = 500, gain: float = 0.6):
    calibrated, log = closed_loop_generate(device, n_bits, block, gain)
    h = conservative(calibrated)
    return extractor.extract(calibrated, h), calibrated, h, log
