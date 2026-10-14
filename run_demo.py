"""
run_demo.py — start the live demo.   Usage:  python run_demo.py

Why a launcher instead of `streamlit run app.py`?
Qiskit 2.x (Rust-backed) can crash with a segmentation fault if it is first
imported inside a thread that later exits and is then used from another
thread. Streamlit does exactly that: the first page load imports Qiskit in a
worker thread, and every button click runs the script in a new thread.
Importing Qiskit here, in the main thread, before Streamlit starts, avoids it.
"""

import sys

import qiskit            # noqa: F401  — must be imported in the main thread
import qiskit_aer        # noqa: F401
from streamlit.web import cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py", *sys.argv[1:]]
    sys.exit(stcli.main())
