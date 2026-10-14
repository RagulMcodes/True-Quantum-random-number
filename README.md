# Quantum Random Number Generator: Live Demo

## Run it (laptop is fine, no GPU needed)
Unzip, open a terminal **inside the `qrng` folder**, then:
```
pip install -r requirements.txt
python run_demo.py
```
Your browser opens at http://localhost:8501. `streamlit run app.py` also works: `qiskit_keeper.py` prevents a Qiskit crash under Streamlit (the file explains why).

## Share it online so teammates can open it

**Option A: Streamlit Community Cloud (free, a permanent link).**
1. Create a GitHub repository and upload everything in this folder, **including `results/`, `figures/` and `.streamlit/`** (tab ⑤ reads them).
2. Go to share.streamlit.io, sign in with GitHub, and click **Create app**.
3. Pick the repository and branch; set the main file to `app.py`. In **Advanced settings**, choose Python 3.12.
4. Click **Deploy**. The first build installs Qiskit and takes a few minutes. You then get a `https://<name>.streamlit.app` link.
5. Free apps go to sleep after a few days without visits. Anyone opening the link can wake it up, which takes about a minute, so open it yourself before the demo.

**Option B: Hugging Face Spaces (free, does not sleep as quickly).** Create a Space with the Streamlit template, upload these files, and keep `app.py` as the app file.

**Option C: share from your laptop for a quick session (no account needed).** Run the app, then in a second terminal run `cloudflared tunnel --url http://localhost:8501` (install cloudflared first). It prints a temporary public `https://…trycloudflare.com` link that works only while your laptop and both commands are running.

On the same Wi-Fi you can also try the **Network URL** that Streamlit prints when it starts, though campus Wi-Fi often blocks device-to-device traffic.

The results for tab ⑤ are already included. To regenerate them (about 30 s):
```
python run_correction_experiment.py
python make_figures.py
```

**Before the demo:** start the app once at home so the first launch doesn't happen in front of your teacher. Close other heavy apps. Keep this README open.

---

## Demo script (about 8 minutes)

**① Quantum coin (1 min)**
- *Say:* "One qubit, a Hadamard gate, a measurement. Quantum mechanics says the result is 0 or 1 with probability ½, and nothing decides it beforehand."
- *Click:* **Measure the qubit** at 10,000. Point to P(1) ≈ 0.5, entropy ≈ 1.0, and the chi-square test passing.
- *Click:* **Roll** a few times, then **Roll 6,000 quantum vs 6,000 Python dice**.
- *Say:* "Both pass the tests. So why bother with quantum?" This sets up tab ②.

**② Can you predict it? (2 min) ← the wow moment**
- *Say:* "Python's `random` uses the Mersenne Twister. It has 624 internal numbers, and every output can be reversed back into one of them."
- *Click:* **Observe 624 numbers from Python random**. The attacker's predictions for the next 20 dice rolls appear.
- *Click:* **Roll the real dice**. Result: **20/20 correct**.
- Do the same for the QRNG. Result: about 3/20, which is chance level.
- *Say:* "Statistical tests can't see this. Predictability is the real difference, and a QRNG has no internal state to steal."

**③ Broken hardware (1.5 min)**
- *Do first:* set **Readout memory to 0** and **Drift to 0**, then generate bits. The measured P(1) matches the formula sin²((π/2+ε)/2) (≈ 0.604 at 12°).
- Drag **ε** to 25° and point at the Bloch arrow tipping toward |1⟩.
- Set it back to ε = 12°, drift = 10°, readout memory = 0.25, and generate again. Every test fails.
- *Say:* "This models real faults: a miscalibrated pulse, calibration drift, and readout afterpulsing that makes consecutive bits correlated."

**④ Fixing it (2 min)**
- Open **How our method works** and explain both stages in one sentence each.
- *Click:* **Run the faulty device and apply every correction**.
- Point out that **Von Neumann gets the bias right but fails the correlation tests**, which is a known limitation (Abbott & Calude 2012). Toeplitz passes but keeps only about 42% of bits. **Ours passes and keeps about 71%.**
- Show the left chart: "The controller never sees the error directly. It infers it from the bits with θ̂ = 2·arcsin(√p̂) and corrects the gate angle every 500 bits."

**⑤ Evidence (1 min)**
- *Say:* "One run could be luck, so here are 20 independent trials per scenario, judged by NIST's acceptance rule." Read the D column.
- Finish with the honest limits paragraph. Examiners respect it.

---

## Likely questions and answers

**"Isn't the simulator using a pseudo-random generator underneath?"**
Yes. Aer samples measurement outcomes with a classical PRNG, so this demo *models* a QRNG. It doesn't produce certified quantum randomness. Running the same circuit on real IBM hardware would give physical randomness, and that's the planned next step.

**"Why does P(1) in tab ③ differ from the formula?"**
Readout memory adds extra 1s on top of the rotation error. Set it to 0 and the measured value matches the formula.

**"Why does the controller's correction overshoot the true error in tab ④?"**
It steers P(1) to 0.5, so it also cancels the bias that readout adds. Its goal is unbiased output, not estimating ε exactly.

**"Why not just use Von Neumann?"**
It assumes independent bits. When readout makes bits correlated, it turns positive correlation into negative correlation (see `figures/fig3_autocorrelation.png`). It also throws away about 80% of the bits.

**"Is Peres better than yours?"**
When bits are independent (scenarios A and B), yes, Peres keeps about the same number of bits. But it fails when bits are correlated (C and D).

**"Is 25% readout memory realistic?"**
It's exaggerated so the effect is visible within 20,000 bits. Real afterpulsing is usually a few percent, and the method works the same way.

---

## Files
| File | Purpose |
|---|---|
| `run_demo.py` | Launches the demo (use this) |
| `app.py` | The live demo app |
| `device.py` | Simulated faulty QRNG with hidden errors (Qiskit Aer) |
| `calibration.py` | **Our method**: closed-loop self-calibration plus Toeplitz extraction |
| `extractors.py` | Von Neumann, Peres, Toeplitz |
| `entropy.py` | Shannon entropy; NIST SP 800-90B MCV and Markov min-entropy |
| `stats_tests.py` | Monobit chi-square, runs, autocorrelation, NIST serial test |
| `mt_attack.py` | Clones Python's Mersenne Twister from 624 outputs |
| `run_correction_experiment.py`, `make_figures.py` | The 20-trial study and its figures |
