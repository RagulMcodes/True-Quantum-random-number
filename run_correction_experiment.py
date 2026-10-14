"""
run_correction_experiment.py — Step 1: bias/correlation scenarios vs correction methods.

For each scenario and each of K independent trials:
  * the device produces N raw bits with the naive Hadamard angle (open loop)
  * VN, Peres and Toeplitz post-process those raw bits
  * an identical device (same hidden drift) runs our closed-loop calibration,
    reported both alone ("Calibration only") and with the extractor ("Ours")

Outputs:  results/summary.csv, results/trials.csv, figures/*.png
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from calibration import closed_loop_generate
from device import HADAMARD_ANGLE, SCENARIOS, FaultyQRNGDevice
from entropy import conservative, markov_min_entropy, mcv_min_entropy, shannon
from extractors import ToeplitzExtractor, peres, von_neumann
from stats_tests import ALPHA, TESTS, lag1_autocorr, run_all

N_RAW = 20_000
K_TRIALS = 20
METHODS = ["Raw", "Von Neumann", "Peres", "Toeplitz", "Calibration only", "Ours"]

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def describe(bits: np.ndarray, n_in: int) -> dict:
    row = {"n_out": len(bits), "efficiency": len(bits) / n_in}
    if len(bits) < 100:
        return row
    row.update({
        "p1": float(np.mean(bits)),
        "abs_bias": abs(float(np.mean(bits)) - 0.5),
        "shannon": shannon(bits),
        "h_mcv": mcv_min_entropy(bits),
        "h_markov": markov_min_entropy(bits),
        "lag1": lag1_autocorr(bits),
    })
    for name, p in run_all(bits).items():
        row[f"p::{name}"] = p
    return row


def main():
    extractor = ToeplitzExtractor(n=1024, security_bits=32)
    rows = []
    traces = {}          # one example trial per scenario, for the drift plot
    t0 = time.time()
    for si, (scen, params) in enumerate(SCENARIOS.items()):
        for trial in range(K_TRIALS):
            seed = 1000 * trial + si   # stable, reproducible
            dev_open = FaultyQRNGDevice(seed=seed, **params)
            dev_loop = FaultyQRNGDevice(seed=seed, **params)   # same hidden drift

            raw = dev_open.run(HADAMARD_ANGLE, N_RAW)
            h_raw = conservative(raw)
            calibrated, log = closed_loop_generate(dev_loop, N_RAW)
            h_cal = conservative(calibrated)

            outputs = {
                "Raw": raw,
                "Von Neumann": von_neumann(raw),
                "Peres": peres(raw),
                "Toeplitz": extractor.extract(raw, h_raw),
                "Calibration only": calibrated,
                "Ours": extractor.extract(calibrated, h_cal),
            }
            for m, bits in outputs.items():
                rows.append({"scenario": scen, "trial": trial, "method": m,
                             "h_used": h_raw if m == "Toeplitz" else h_cal if m == "Ours" else np.nan,
                             **describe(bits, N_RAW)})
            if trial == 0:
                traces[scen] = dict(raw=raw, calibrated=calibrated, log=log,
                                    eps_log=dev_open.eps_log)
        print(f"  {scen:26s} done  ({time.time() - t0:5.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv("results/trials.csv", index=False)

    # ---- aggregate --------------------------------------------------------
    pcols = [c for c in df.columns if c.startswith("p::")]
    for c in pcols:
        df[c.replace("p::", "pass::")] = df[c] > ALPHA
    df["pass::All"] = df[[c.replace("p::", "pass::") for c in pcols]].all(axis=1)
    agg = df.groupby(["scenario", "method"], sort=False).agg(
        efficiency=("efficiency", "mean"),
        abs_bias=("abs_bias", "mean"),
        lag1=("lag1", "mean"),
        h_mcv=("h_mcv", "mean"),
        h_markov=("h_markov", "mean"),
        **{c.replace("pass::", "pass_"): (c, "mean")
           for c in df.columns if c.startswith("pass::")},
    ).reset_index()
    agg.to_csv("results/summary.csv", index=False)
    np.save("results/traces.npy", traces, allow_pickle=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    show = agg[["scenario", "method", "efficiency", "abs_bias", "lag1",
                "h_markov", "pass_Monobit χ²", "pass_Runs", "pass_Autocorr",
                "pass_Serial", "pass_All"]]
    print(show.round(4).to_string(index=False))
    print(f"\nNIST acceptance for K={K_TRIALS}: pass proportion >= "
          f"{0.99 - 3*np.sqrt(0.99*0.01/K_TRIALS):.3f}")


if __name__ == "__main__":
    main()
