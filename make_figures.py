"""make_figures.py — figures for Step 1 (reads results/ written by the experiment)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extractors import ToeplitzExtractor, peres, von_neumann
from entropy import conservative
from stats_tests import ALPHA

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"Ours": "#2a78d6", "Von Neumann": "#eb6834", "Peres": "#1baf7a",
          "Toeplitz": "#eda100", "Calibration only": "#e87ba4", "Raw": "#9b9a95"}
ORDER = ["Raw", "Von Neumann", "Peres", "Toeplitz", "Calibration only", "Ours"]
K = 20
NIST_MIN = 0.99 - 3 * np.sqrt(0.99 * 0.01 / K)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 11,
})

agg = pd.read_csv("results/summary.csv")
traces = np.load("results/traces.npy", allow_pickle=True).item()
scens = [s for s in agg.scenario.unique() if s != "Ideal"]


# ── Figure 1: pass rate and efficiency, grouped by scenario ────────────────
def fig_tradeoff():
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    x = np.arange(len(scens))
    w = 0.13
    for i, m in enumerate(ORDER):
        sub = agg[agg.method == m].set_index("scenario").loc[scens]
        off = (i - (len(ORDER) - 1) / 2) * w
        for ax, col in zip(axes, ["pass_All", "efficiency"]):
            vals = sub[col].values
            bars = ax.bar(x + off, vals, w * 0.9, color=COLORS[m], label=m,
                          edgecolor=SURFACE, linewidth=1)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                        f"{v:.0%}" if col == "pass_All" else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7, color=INK2)
    axes[0].axhline(NIST_MIN, color=INK, ls="--", lw=1)
    axes[0].text(len(scens) - 0.45, 1.11, f"dashed line = NIST acceptance threshold ({NIST_MIN:.0%})",
                 ha="right", fontsize=8, color=INK)
    axes[0].set_ylabel("Trials passing ALL 4 tests")
    axes[0].set_title("Quality: share of 20 trials passing monobit, runs, autocorrelation and serial tests (α = 0.01)", loc="left")
    axes[0].set_ylim(0, 1.15)
    axes[1].set_ylabel("Output bits per raw bit")
    axes[1].set_title("Efficiency: how many usable bits survive correction", loc="left")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_xticks(x, scens)
    axes[0].legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.28),
                   frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/fig1_quality_vs_efficiency.png", dpi=160)
    plt.close(fig)


# ── Figure 2: closed-loop calibration tracking the hidden drift ────────────
def fig_tracking():
    tr = traces["B: Drifting bias"]
    eps_t, eps = zip(*tr["eps_log"])
    log_t, cmd = zip(*tr["log"])
    est_eps = np.degrees(np.pi / 2 - np.array(cmd))          # controller's belief
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)

    ax = axes[0]
    ax.plot(eps_t, np.degrees(eps), color=COLORS["Raw"], lw=2, label="True hidden error ε(t)")
    ax.step(log_t, est_eps, where="pre", color=COLORS["Ours"], lw=2,
            label="Controller's correction (estimated ε)")
    ax.set_ylabel("Rotation error (degrees)")
    ax.set_title("Self-calibration: the controller tracks a drifting error it cannot see directly", loc="left")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.09, 0.0))

    ax = axes[1]
    W = 500
    for key, m, lbl in [("raw", "Raw", "Open loop (naive H)"),
                        ("calibrated", "Ours", "Closed loop (self-calibrated)")]:
        b = tr[key].astype(float)
        roll = np.convolve(b, np.ones(W) / W, mode="valid")
        ax.plot(np.arange(len(roll)) + W, roll, color=COLORS[m], lw=2, label=lbl)
    ax.axhline(0.5, color=INK, ls="--", lw=1)
    ax.text(200, 0.505, "ideal 0.5", fontsize=8, color=INK)
    ax.set_ylim(0.38, 0.75)
    ax.set_ylabel(f"P(1), rolling {W}-bit window")
    ax.set_xlabel("Shot index (time)")
    ax.set_title("Result: bias removed at the source, before any post-processing", loc="left")
    ax.legend(frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig("figures/fig2_calibration_tracking.png", dpi=160)
    plt.close(fig)


# ── Figure 3: autocorrelation by lag — why VN/Peres fail on correlated data ─
def autocorr(bits, lag):
    x = bits.astype(float) * 2 - 1
    x -= x.mean()
    return np.dot(x[:-lag], x[lag:]) / np.dot(x, x)


def fig_autocorr():
    tr = traces["D: Drift + correlation"]
    T = ToeplitzExtractor()
    outs = {"Raw": tr["raw"], "Von Neumann": von_neumann(tr["raw"]),
            "Peres": peres(tr["raw"]),
            "Ours": T.extract(tr["calibrated"], conservative(tr["calibrated"]))}
    lags = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(12, 5))
    for m, b in outs.items():
        r = [autocorr(b, L) for L in lags]
        ax.plot(lags, r, color=COLORS[m], lw=2, marker="o", ms=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=f"{m}  (n={len(b):,})")
        ax.text(10.15, r[-1], m, va="center", fontsize=9, color=INK2)
        ax.text(1, r[0] + (0.006 if m != "Von Neumann" else -0.012),
                f"{r[0]:+.3f}", ha="center", fontsize=8, color=INK2)
    n_min = min(len(b) for b in outs.values())
    band = 2.576 / np.sqrt(n_min)
    ax.axhspan(-band, band, color=GRID, alpha=0.6, lw=0)
    ax.text(5.5, band + 0.003, "99% band for independent bits (smallest n)", ha="center",
            fontsize=8, color=INK2)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(lags)
    ax.set_xlim(0.6, 11)
    ax.set_ylim(-0.07, 0.13)
    ax.set_xlabel("Lag (bits)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Scenario D: VN turns positive correlation into negative correlation; ours removes it", loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig("figures/fig3_autocorrelation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_tradeoff()
    fig_tracking()
    fig_autocorr()
    print("figures written")
