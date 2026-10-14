"""
qiskit_keeper.py — import this BEFORE qiskit in any Streamlit page.

Qiskit 2.x (Rust core) can segfault if it is first imported in a thread that
later exits and is then used from another thread. Streamlit runs every page
rerun in a fresh thread, so a plain `import qiskit` in app.py crashes the
server on the second click.

Fix: do the first import in a daemon thread that never exits. Python caches
modules, so this runs once per server process; every later
`from qiskit import ...` just reuses the already-loaded module.
This makes `streamlit run app.py` safe, which hosted platforms
(Streamlit Community Cloud, Hugging Face Spaces) require.
"""

import threading

_ready = threading.Event()


def _keep():
    import qiskit       # noqa: F401
    import qiskit_aer   # noqa: F401
    _ready.set()
    threading.Event().wait()     # stay alive for the life of the process


threading.Thread(target=_keep, name="qiskit-keeper", daemon=True).start()
_ready.wait()
