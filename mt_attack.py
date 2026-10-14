"""
mt_attack.py — Cloning Python's `random` (Mersenne Twister MT19937).

Python's generator keeps 624 x 32-bit words of internal state. Each output
is one state word passed through a reversible "tempering" function. So:
  1. observe 624 consecutive 32-bit outputs
  2. invert the tempering on each  ->  the full internal state
  3. load that state into a fresh generator  ->  it now predicts every
     future output of the original, exactly, forever.
No seed is needed. This is why MT19937 must never be used for security
(Python's own docs say so: use `secrets` instead).

Running the same attack on QRNG output fails, because there is no internal
state to recover: each bit is a fresh quantum measurement.
"""

from __future__ import annotations

import random

N_STATE = 624


def untemper(y: int) -> int:
    y ^= y >> 18
    y ^= (y << 15) & 0xEFC60000
    x = y
    for _ in range(5):
        x = y ^ ((x << 7) & 0x9D2C5680)
    y = x & 0xFFFFFFFF
    x = y
    for _ in range(3):
        x = y ^ (x >> 11)
    return x & 0xFFFFFFFF


def clone_from_outputs(outputs: list[int]) -> random.Random:
    if len(outputs) < N_STATE:
        raise ValueError("need 624 consecutive 32-bit outputs")
    state = tuple(untemper(o) for o in outputs[-N_STATE:]) + (N_STATE,)
    clone = random.Random()
    clone.setstate((3, state, None))
    return clone


def bits_to_words(bits, width: int = 32) -> list[int]:
    words = []
    for i in range(0, len(bits) - width + 1, width):
        w = 0
        for b in bits[i:i + width]:
            w = (w << 1) | int(b)
        words.append(w)
    return words
