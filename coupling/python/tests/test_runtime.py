"""Process-level behaviour of the embedded Julia runtime."""
import concurrent.futures as cf
import os
import signal
import subprocess
import sys
import textwrap
import time

import numpy as np

import ace_jax_coupling as ajc


def test_import_does_not_load_library():
    code = "import sys, ace_jax_coupling; from ace_jax_coupling import _loader; sys.exit(0 if _loader._LIB is None else 1)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_threads_concurrent_calls(cases):
    c = cases["CrMnFe_o3d6"]
    ref = ajc.couple_raw(c["mb"], c["R"], c["Y"])
    with cf.ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda _: ajc.couple_raw(c["mb"], c["R"], c["Y"]), range(8)))
    assert all(np.array_equal(o.A2B_vals, ref.A2B_vals) and np.array_equal(o.aa_idx, ref.aa_idx) for o in outs)


def test_sigint_still_raises_keyboardinterrupt(cases, tmp_path):
    """Loading the library must not take over SIGINT (built with handle-signals=no)."""
    c = cases["tiny"]
    script = textwrap.dedent(f"""
        import sys, time, ace_jax_coupling as ajc
        ajc.couple_raw({c['mb']!r}, {c['R']!r}, {c['Y']!r})
        print("ready", flush=True)
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("kbi", flush=True); sys.exit(7)
    """)
    p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, env=os.environ)
    assert p.stdout.readline().strip() == "ready"
    time.sleep(0.2)
    p.send_signal(signal.SIGINT)
    out, _ = p.communicate(timeout=20)
    assert p.returncode == 7 and "kbi" in out
