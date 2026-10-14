"""
app.py — Live demo: Quantum Random Number Generator with imperfect hardware.

Run:   python run_demo.py   or   streamlit run app.py   (both safe; see qiskit_keeper.py)
Everything on screen is generated live by the Qiskit Aer simulator.
"""

from __future__ import annotations

import os
import random

import qiskit_keeper  # noqa: F401  — must come before any qiskit import

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from calibration import closed_loop_generate
from device import HADAMARD_ANGLE, FaultyQRNGDevice
from entropy import conservative, markov_min_entropy, shannon
from extractors import ToeplitzExtractor, peres, von_neumann
from mt_attack import N_STATE, bits_to_words, clone_from_outputs
from stats_tests import ALPHA, TESTS, lag1_autocorr, monobit_chi2, run_all

# ── look & feel ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quantum RNG Demo", page_icon="🎲", layout="wide")
C = {"Ours": "#2a78d6", "Von Neumann": "#eb6834", "Peres": "#1baf7a",
     "Toeplitz": "#eda100", "Calibration only": "#e87ba4", "Raw": "#9b9a95",
     "Quantum": "#2a78d6", "Classical": "#eb6834"}
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.dice {font-size: 120px; line-height: 1; text-align: center;}
.bits {font-family: ui-monospace, monospace; font-size: 15px; letter-spacing: 2px;
       word-break: break-all; padding: 12px 14px; border-radius: 8px;
       background: rgba(128,128,128,0.08);}
.big {font-size: 28px; font-weight: 700;}
.ok  {color: #1b8a3c; font-weight: 600;}
.bad {color: #c62828; font-weight: 600;}
</style>""", unsafe_allow_html=True)
DICE = "⚀⚁⚂⚃⚄⚅"

SIM = AerSimulator()


def plot_layout(fig, h=330, **kw):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.12, yanchor="top", x=0), **kw)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    fig.update_xaxes(showgrid=False)
    return fig


def verdict(p):
    return "✓ pass" if p > ALPHA else "✗ fail"


# ── quantum primitives (live Aer runs) ───────────────────────────────────────
def _circuit(n: int) -> QuantumCircuit:
    # Built fresh per call so no circuit object is shared between Streamlit's
    # script threads. (Launch via run_demo.py — see the note there.)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.measure_all()
    return qc


@st.cache_data
def circuit_diagram() -> str:
    return str(_circuit(1).draw("text"))


def quantum_ints(n_qubits: int, shots: int) -> list[int]:
    mem = SIM.run(_circuit(n_qubits), shots=shots, memory=True).result().get_memory()
    return [int(s, 2) for s in mem]


def quantum_bits(n: int) -> np.ndarray:
    return np.array(quantum_ints(1, n), dtype=np.uint8)


def quantum_dice(n: int) -> list[int]:
    out = []
    while len(out) < n:
        out += [v + 1 for v in quantum_ints(3, 2 * n) if v <= 5]   # reject 6, 7
    return out[:n]


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎲 Quantum RNG")
    st.markdown("**Randomness from quantum measurement, made robust to imperfect hardware.**")
    st.markdown("---")
    st.markdown("**Ragul** · Quantum Computing project")
    st.markdown("Stack: Qiskit 2 · Qiskit Aer · NumPy/SciPy · Streamlit")
    st.markdown("---")
    st.caption("Every number in this demo is produced live by a quantum circuit "
               "running on the Qiskit Aer simulator.")

st.title("Quantum Random Number Generator")
tabs = st.tabs(["① Quantum coin", "② Can you predict it?", "③ Broken hardware",
                "④ Fixing it", "⑤ Evidence"])

# ═════════════════════════════════════════════════════════════════════════════
# ① QUANTUM COIN
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.subheader("One qubit, one gate, one measurement")
        st.code(circuit_diagram(), language=None)
        st.latex(r"H|0\rangle=\tfrac{1}{\sqrt2}\left(|0\rangle+|1\rangle\right)"
                 r"\;\Rightarrow\;P(0)=P(1)=\tfrac12")
        st.markdown("The outcome isn't decided until the qubit is measured. "
                    "No hidden state, no seed, no formula.")
        n = st.select_slider("Number of measurements", [100, 1000, 10_000, 100_000], 1000)
        if st.button("⚛️ Measure the qubit", type="primary", width="stretch"):
            st.session_state.coin = quantum_bits(n)
    with right:
        bits = st.session_state.get("coin")
        if bits is None:
            st.info("Press **Measure the qubit** to generate random bits.")
        else:
            st.markdown(f'<div class="bits">{"".join(map(str, bits[:320]))}'
                        f'{"…" if len(bits) > 320 else ""}</div>', unsafe_allow_html=True)
            n1 = int(bits.sum()); n0 = len(bits) - n1; p = monobit_chi2(bits)
            m1, m2, m3 = st.columns(3)
            m1.metric("P(1)", f"{n1 / len(bits):.4f}", help="ideal 0.5000")
            m2.metric("Entropy (bits)", f"{shannon(bits):.4f}", help="Shannon entropy per bit, max 1.0")
            m3.metric("χ² p-value", f"{p:.3f}", verdict(p), delta_color="off")
            fig = go.Figure(go.Bar(x=["0", "1"], y=[n0, n1], marker_color=C["Quantum"],
                                   text=[n0, n1], textposition="outside", width=0.5))
            fig.add_hline(y=len(bits) / 2, line_dash="dash", annotation_text="expected")
            fig.update_xaxes(type="category")
            st.plotly_chart(plot_layout(fig, 260, title="Outcome counts"), width="stretch")

    st.markdown("---")
    d1, d2, d3 = st.columns([1, 1, 1.6], gap="large")
    with d1:
        st.subheader("Quantum dice")
        st.caption("3 qubits → 0–7. Reject 6 and 7, map 0–5 → 1–6 (no modulo bias).")
        if st.button("🎲 Roll", width="stretch"):
            st.session_state.die = quantum_dice(1)[0]
        die = st.session_state.get("die")
        if die:
            st.markdown(f'<div class="dice">{DICE[die - 1]}</div>'
                        f'<p style="text-align:center" class="big">{die}</p>', unsafe_allow_html=True)
    with d2:
        st.subheader("8-bit number")
        st.caption("8 qubits measured at once → 0–255.")
        if st.button("Generate byte", width="stretch"):
            st.session_state.byte = quantum_ints(8, 1)[0]
        b = st.session_state.get("byte")
        if b is not None:
            st.markdown(f'<p class="big" style="font-size:64px;text-align:center">{b}</p>'
                        f'<p style="text-align:center;font-family:monospace">{b:08b}</p>',
                        unsafe_allow_html=True)
    with d3:
        st.subheader("Is the dice fair?")
        if st.button("Roll 6,000 quantum vs 6,000 Python dice", width="stretch"):
            st.session_state.dice_cmp = (quantum_dice(6000),
                                         [random.randint(1, 6) for _ in range(6000)])
        dc = st.session_state.get("dice_cmp")
        if dc:
            from scipy.stats import chisquare
            fig = go.Figure()
            for name, rolls in zip(["Quantum", "Classical"], dc):
                cnt = np.bincount(rolls, minlength=7)[1:]
                p = chisquare(cnt).pvalue
                fig.add_bar(x=list(range(1, 7)), y=cnt, name=f"{name}  (χ² p = {p:.2f})",
                            marker_color=C[name])
            fig.add_hline(y=1000, line_dash="dash")
            st.plotly_chart(plot_layout(fig, 280, barmode="group"), width="stretch")
            st.caption("Both pass. **Statistical tests can't tell quantum from classical.** "
                       "Tab ② shows the real difference.")

# ═════════════════════════════════════════════════════════════════════════════
# ② PREDICTION ATTACK
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Python's `random` passes every test. It's still predictable.")
    st.markdown(
        "Python uses the **Mersenne Twister**: 624 internal numbers + a formula. "
        "Each output is one internal number, scrambled in a *reversible* way. "
        "So watching **624 outputs** reveals the whole internal state, and from then on "
        "every future 'random' number can be predicted. No seed needed.")
    ca, cb = st.columns(2, gap="large")

    def attack_panel(col, key, label, source):
        with col:
            st.markdown(f"### {label}")
            if st.button(f"1. Observe 624 numbers from {label}", key=f"obs_{key}",
                         width="stretch"):
                st.session_state[key] = source()
                st.session_state.pop(f"rev_{key}", None)
            s = st.session_state.get(key)
            if not s:
                return
            st.markdown('<div class="bits">' + ", ".join(str(x) for x in s["observed"][:8]) +
                        f", … ({N_STATE} numbers)</div>", unsafe_allow_html=True)
            st.markdown("**2. Attacker clones the generator and predicts the next 20 dice rolls:**")
            st.markdown(f'<p style="font-size:34px">{" ".join(DICE[d - 1] for d in s["pred"])}</p>',
                        unsafe_allow_html=True)
            if st.button("3. Roll the real dice", key=f"rev_btn_{key}", type="primary",
                         width="stretch"):
                st.session_state[f"rev_{key}"] = True
            if st.session_state.get(f"rev_{key}"):
                hits = [p == a for p, a in zip(s["pred"], s["actual"])]
                row = " ".join(f'<span class="{"ok" if h else "bad"}">{DICE[a - 1]}</span>'
                               for a, h in zip(s["actual"], hits))
                st.markdown(f'<p style="font-size:34px">{row}</p>', unsafe_allow_html=True)
                k = sum(hits)
                st.markdown(f'<p class="big {"bad" if k > 10 else "ok"}">{k}/20 predicted correctly</p>',
                            unsafe_allow_html=True)
                st.caption("Random guessing gets about 3/20 (1 in 6).")

    def python_source():
        victim = random.Random(int.from_bytes(os.urandom(8), "big"))   # unknown seed
        observed = [victim.getrandbits(32) for _ in range(N_STATE)]
        clone = clone_from_outputs(observed)
        return {"observed": observed, "pred": [clone.randint(1, 6) for _ in range(20)],
                "actual": [victim.randint(1, 6) for _ in range(20)]}

    def quantum_source():
        observed = bits_to_words(quantum_bits(N_STATE * 32))
        clone = clone_from_outputs(observed)
        return {"observed": observed, "pred": [clone.randint(1, 6) for _ in range(20)],
                "actual": quantum_dice(20)}

    attack_panel(ca, "atk_py", "Python random", python_source)
    attack_panel(cb, "atk_q", "the QRNG", quantum_source)
    st.info("**Takeaway:** a PRNG only *looks* random. A QRNG has no internal state "
            "to steal, so the same attack falls to chance level.")

# ═════════════════════════════════════════════════════════════════════════════
# ③ BROKEN HARDWARE
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Real quantum hardware is imperfect")
    s1, s2, s3 = st.columns(3)
    eps = s1.slider("Over-rotation error ε (°)", 0.0, 30.0, 12.0, 0.5,
                    help="Ry(π/2 + ε) applied instead of a perfect H. Miscalibrated pulse.")
    drift = s2.slider("Drift amplitude (°)", 0.0, 15.0, 10.0, 0.5,
                      help="ε wanders slowly over time (calibration goes stale).")
    after = s3.slider("Readout memory (afterpulsing)", 0.0, 0.4, 0.25, 0.01,
                      help="After a '1', a true 0 is misread as 1 with this probability → correlated bits.")
    st.session_state.dev = dict(eps_deg=eps, drift_amp_deg=drift,
                                drift_walk_deg=0.15 if drift else 0.0, afterpulse=after)

    bl, br = st.columns([1, 1.2], gap="large")
    with bl:
        th = np.radians(90 + eps)
        a = np.linspace(0, 2 * np.pi, 200)
        fig = go.Figure(go.Scatter(x=np.sin(a), y=np.cos(a), mode="lines", hoverinfo="skip",
                                   line=dict(color="rgba(128,128,128,0.5)", width=2), showlegend=False))
        fig.add_scatter(x=[-1.1, 1.1], y=[0, 0], mode="lines", hoverinfo="skip", showlegend=False,
                        line=dict(color="rgba(128,128,128,0.3)", width=1, dash="dot"))
        fig.add_scatter(x=[0, 0], y=[-1.1, 1.1], mode="lines", hoverinfo="skip", showlegend=False,
                        line=dict(color="rgba(128,128,128,0.3)", width=1, dash="dot"))
        for name, ang, col in [("Ideal: H|0⟩ = |+⟩  →  P(1) = 0.5", np.pi / 2, "#1baf7a"),
                               (f"Faulty: Ry(π/2 + {eps:.1f}°)|0⟩", th, "#eb6834")]:
            x, z = np.sin(ang), np.cos(ang)
            fig.add_annotation(x=x, y=z, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowwidth=4, arrowcolor=col)
            fig.add_scatter(x=[x], y=[z], mode="markers", marker=dict(size=10, color=col), name=name,
                            showlegend=False)
        zf = np.cos(th)
        fig.add_scatter(x=[np.sin(th), 0], y=[zf, zf], mode="lines", showlegend=False, hoverinfo="skip",
                        line=dict(color="#eb6834", width=1.5, dash="dash"))
        for (x, y, t) in [(0, 1.22, "|0⟩"), (0, -1.22, "|1⟩"), (1.25, 0, "|+⟩"), (-1.25, 0, "|−⟩")]:
            fig.add_annotation(x=x, y=y, text=t, showarrow=False, font=dict(size=18))
        fig.update_xaxes(visible=False, range=[-1.35, 1.35], scaleanchor="y")
        fig.update_yaxes(visible=False, range=[-1.35, 1.35])
        fig = plot_layout(fig, 400, title="Bloch sphere (x–z slice): the state tips toward |1⟩")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
        st.markdown('<span style="color:#1baf7a">●</span> Ideal: H|0⟩ = |+⟩, on the equator → P(1) = 0.5'
                    f'&nbsp;&nbsp;&nbsp;<span style="color:#eb6834">●</span> Faulty: Ry(π/2 + {eps:.1f}°)|0⟩. '
                    'The dashed line projects onto the |0⟩–|1⟩ axis: lower means more 1s.',
                    unsafe_allow_html=True)
        p1 = np.sin(th / 2) ** 2
        st.latex(rf"P(1)=\sin^2\!\left(\frac{{\pi/2+\varepsilon}}{{2}}\right)={p1:.4f}")
    with br:
        if st.button("⚛️ Generate 20,000 bits from this faulty device", type="primary",
                     width="stretch"):
            dev = FaultyQRNGDevice(seed=int.from_bytes(os.urandom(4), "big"),
                                   **st.session_state.dev)
            st.session_state.raw = dev.run(HADAMARD_ANGLE, 20_000)
        raw = st.session_state.get("raw")
        if raw is None:
            st.info("Set the faults, then generate bits.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("P(1)", f"{raw.mean():.4f}", f"{raw.mean() - 0.5:+.4f} vs ideal", delta_color="off")
            m2.metric("Bit-to-bit correlation", f"{lag1_autocorr(raw):+.3f}", "ideal 0.000", delta_color="off")
            m3.metric("Min-entropy", f"{markov_min_entropy(raw):.3f}", "ideal 1.000", delta_color="off")
            res = run_all(raw)
            st.dataframe(pd.DataFrame({"Test": list(res), "p-value": [f"{p:.2e}" for p in res.values()],
                                       "Result": [verdict(p) for p in res.values()]}),
                         hide_index=True, width="stretch")
            W = 500
            roll = np.convolve(raw.astype(float), np.ones(W) / W, "valid")
            fig = go.Figure(go.Scatter(y=roll[::10], x=np.arange(len(roll))[::10] + W,
                                       line=dict(color=C["Raw"], width=2), name="raw"))
            fig.add_hline(y=0.5, line_dash="dash", annotation_text="ideal")
            st.plotly_chart(plot_layout(fig, 230, title="P(1) over time (500-bit window)",
                                        showlegend=False), width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# ④ FIXING IT
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Six ways to handle the faulty device, compared live")
    dp = st.session_state.get("dev", {})
    st.caption(f"Device settings from tab ③: ε = {dp.get('eps_deg', 12):.1f}°, "
               f"drift = {dp.get('drift_amp_deg', 10):.1f}°, "
               f"readout memory = {dp.get('afterpulse', 0.25):.2f}")
    with st.expander("How our method works", expanded=False):
        st.markdown(
            "**Stage 1: fix the source.** The Born rule gives P(1) = sin²(θ/2), so the device's real "
            "rotation can be read from its own output: θ̂ = 2·arcsin(√p̂). After every 500-bit block we "
            "correct the commanded angle (feedback gain 0.6). No extra calibration shots are needed, "
            "and it follows drift continuously.\n\n"
            "**Stage 2: clean the residue.** Calibration can't remove readout correlation, so the stream "
            "goes through a Toeplitz-hashing extractor. Its output length comes from a conservative "
            "min-entropy estimate (NIST SP 800-90B MCV and Markov), which accounts for correlation.")
    if st.button("⚛️ Run the faulty device and apply every correction", type="primary"):
        seed = int.from_bytes(os.urandom(4), "big")
        dev_a = FaultyQRNGDevice(seed=seed, **dp)
        dev_b = FaultyQRNGDevice(seed=seed, **dp)       # identical hidden faults
        with st.spinner("Running quantum circuits…"):
            raw = dev_a.run(HADAMARD_ANGLE, 20_000)
            cal, log = closed_loop_generate(dev_b, 20_000)
        T = ToeplitzExtractor()
        st.session_state.fix = dict(
            outs={"Raw": raw, "Von Neumann": von_neumann(raw), "Peres": peres(raw),
                  "Toeplitz": T.extract(raw, conservative(raw)),
                  "Calibration only": cal, "Ours": T.extract(cal, conservative(cal))},
            raw=raw, cal=cal, log=log, eps_log=dev_a.eps_log)
    fx = st.session_state.get("fix")
    if fx:
        rows = []
        for m, b in fx["outs"].items():
            res = run_all(b)
            rows.append({"Method": m, "Output bits": len(b), "Kept": f"{len(b) / 20000:.0%}",
                         "P(1)": f"{b.mean():.4f}", "Correlation": f"{lag1_autocorr(b):+.3f}",
                         **{k: verdict(v) for k, v in res.items()},
                         "Verdict": "✅ RANDOM" if all(v > ALPHA for v in res.values()) else "❌ NOT RANDOM"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("One run. Tab ⑤ shows the same comparison repeated 20 times per scenario.")

        g1, g2 = st.columns(2, gap="large")
        with g1:
            t, e = zip(*fx["eps_log"]); lt, c = zip(*fx["log"])
            fig = go.Figure()
            fig.add_scatter(x=t, y=np.degrees(e), name="True hidden error", line=dict(color=C["Raw"], width=3))
            fig.add_scatter(x=lt, y=np.degrees(np.pi / 2 - np.array(c)), name="Controller's correction",
                            line=dict(color=C["Ours"], width=2, shape="vh"))
            fig.update_yaxes(title="degrees")
            st.plotly_chart(plot_layout(fig, 320, title="Self-calibration tracks the hidden error"),
                            width="stretch")
            if dp.get("afterpulse", 0) > 0:
                st.caption("With readout memory on, the controller also cancels the extra 1s caused by "
                           "readout, so its correction sits above the rotation error alone. It steers "
                           "P(1) to 0.5, not ε to its true value.")
        with g2:
            W = 500; fig = go.Figure()
            for key, m, nm in [("raw", "Raw", "Without calibration"), ("cal", "Ours", "With calibration")]:
                r = np.convolve(fx[key].astype(float), np.ones(W) / W, "valid")
                fig.add_scatter(x=np.arange(len(r))[::10] + W, y=r[::10], name=nm,
                                line=dict(color=C[m], width=2))
            fig.add_hline(y=0.5, line_dash="dash")
            st.plotly_chart(plot_layout(fig, 320, title="P(1) over time"), width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# ⑤ EVIDENCE
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Evidence: 4 fault scenarios × 6 methods × 20 independent trials")
    if not os.path.exists("results/summary.csv"):
        st.warning("Run `python run_correction_experiment.py` and `python make_figures.py` first.")
    else:
        agg = pd.read_csv("results/summary.csv")
        piv = agg[agg.scenario != "Ideal"].pivot(index="method", columns="scenario", values="pass_All")
        eff = agg[agg.scenario != "Ideal"].pivot(index="method", columns="scenario", values="efficiency")
        order = ["Raw", "Von Neumann", "Peres", "Toeplitz", "Calibration only", "Ours"]
        tbl = pd.DataFrame({sc: [f"{piv.loc[m, sc]:.0%} pass · {eff.loc[m, sc]:.2f} kept" for m in order]
                            for sc in piv.columns}, index=order)
        st.dataframe(tbl, width="stretch")
        st.caption("NIST acceptance for 20 trials: ≥ 92% of trials must pass (α = 0.01).")
        for f in ["fig1_quality_vs_efficiency.png", "fig3_autocorrelation.png",
                  "fig2_calibration_tracking.png"]:
            if os.path.exists(f"figures/{f}"):
                st.image(f"figures/{f}", width="stretch")
        st.markdown(
            "**Honest limits.** Peres matches our efficiency when bits are independent (A, B). "
            "Our advantage is on correlated noise (C, D): 100% pass rate while keeping 1.65× more bits "
            "than Toeplitz alone. Stage 1 needs control of the hardware's rotation angle.")
