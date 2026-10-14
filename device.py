"""
device.py — A simulated *imperfect* quantum random number source.

The device hides its true error parameters. The controller (our code) can
only do what a real experimenter can do: command a rotation angle and read
back measured bits. That separation is what makes the self-calibration
experiment meaningful — the controller never peeks at the true error.

Physical error models
---------------------
1. Over-rotation (miscalibrated pulse)
   The commanded Ry(c) is applied as Ry(c + eps(t)). With c = pi/2 (the
   Hadamard-equivalent rotation on |0>), P(1) = sin^2((pi/2 + eps)/2) != 1/2.

2. Drift
   eps(t) is not constant: control electronics and qubit frequency drift
   slowly over time, so the pulse calibration goes stale. Modelled as a slow
   sinusoid plus a small random walk.

3. Readout memory (correlation)
   After a '1' is read, residual signal in the readout chain (afterpulsing in
   single-photon detectors; residual resonator photons in superconducting
   readout) makes the *next* readout more likely to report '1'. A true 0 is
   misread as 1 with probability `afterpulse` if the previous reported bit
   was 1. This makes consecutive bits positively correlated, which breaks the
   independence assumption that von Neumann debiasing needs.

The quantum part (state preparation + Born-rule sampling) runs on Qiskit Aer.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

_SIM = AerSimulator()


def _ry_circuit(angle: float) -> QuantumCircuit:
    # Fresh circuit per call; Aer runs ry/measure natively, no transpile needed.
    qc = QuantumCircuit(1, 1)
    qc.ry(angle, 0)
    qc.measure(0, 0)
    return qc

HADAMARD_ANGLE = np.pi / 2     # Ry(pi/2)|0> = H|0> = |+>


class FaultyQRNGDevice:
    def __init__(
        self,
        eps_deg: float = 0.0,
        drift_amp_deg: float = 0.0,
        drift_period: int = 20_000,
        drift_walk_deg: float = 0.0,
        afterpulse: float = 0.0,
        chunk: int = 250,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self._eps0 = np.radians(eps_deg)
        self._amp = np.radians(drift_amp_deg)
        self._period = drift_period
        self._phase = self.rng.uniform(0, 2 * np.pi)
        self._walk_step = np.radians(drift_walk_deg)
        self._walk = 0.0
        self._afterpulse = afterpulse
        self._chunk = chunk           # shots per constant-angle segment
        self.t = 0                    # device clock (shot index)
        self._last_bit = 0
        self.eps_log: list[tuple[int, float]] = []   # (t, true eps) — for plots only

    # -- hidden truth -------------------------------------------------------
    def _eps(self, t: int) -> float:
        self._walk += self.rng.normal(0, self._walk_step) if self._walk_step else 0.0
        drift = self._amp * np.sin(2 * np.pi * t / self._period + self._phase)
        return self._eps0 + drift + self._walk

    # -- public interface: command an angle, get bits back -----------------
    def run(self, commanded_angle: float, n_bits: int) -> np.ndarray:
        n_chunks = int(np.ceil(n_bits / self._chunk))
        angles = []
        for i in range(n_chunks):
            e = self._eps(self.t + i * self._chunk)
            self.eps_log.append((self.t + i * self._chunk, e))
            angles.append(commanded_angle + e)

        job = _SIM.run(
            [_ry_circuit(a) for a in angles],
            shots=self._chunk,
            memory=True,
            seed_simulator=int(self.rng.integers(2**31)),
        )
        res = job.result()
        true_bits = np.fromiter(
            (int(b) for i in range(n_chunks) for b in res.get_memory(i)),
            dtype=np.uint8,
        )[:n_bits]

        out = self._readout(true_bits)
        self.t += n_bits
        return out

    def _readout(self, true_bits: np.ndarray) -> np.ndarray:
        if self._afterpulse == 0:
            return true_bits
        out = np.empty_like(true_bits)
        flips = self.rng.random(len(true_bits)) < self._afterpulse
        last = self._last_bit
        for i, b in enumerate(true_bits):
            r = 1 if (b == 1 or (last == 1 and flips[i])) else 0
            out[i] = r
            last = r
        self._last_bit = last
        return out


# Scenario presets used throughout the project ------------------------------
SCENARIOS = {
    "Ideal":                 dict(eps_deg=0.0),
    "A: Constant bias":      dict(eps_deg=12.0),
    "B: Drifting bias":      dict(eps_deg=12.0, drift_amp_deg=10.0, drift_walk_deg=0.15),
    "C: Correlated readout": dict(eps_deg=12.0, afterpulse=0.25),
    "D: Drift + correlation": dict(eps_deg=12.0, drift_amp_deg=10.0,
                                   drift_walk_deg=0.15, afterpulse=0.25),
}
