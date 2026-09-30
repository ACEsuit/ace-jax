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


_IMAGES = r"""
import ctypes, hashlib, json, sys
import ace_jax_coupling as a
from ace_jax_coupling import _loader
c = json.load(open(sys.argv[1]))["tiny"]
a.couple_raw(c["mb"], c["R"], c["Y"])
root = str(_loader.bundle_root())
if sys.platform == "darwin":
    d = ctypes.CDLL(None)
    d._dyld_get_image_name.restype = ctypes.c_char_p
    paths = [d._dyld_get_image_name(i).decode() for i in range(d._dyld_image_count())]
else:
    paths = [l.split()[-1] for l in open("/proc/self/maps") if len(l.split()) >= 6]
paths = sorted({p for p in paths if p.startswith(root)})
print(json.dumps({p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in paths}))
"""


def test_no_library_loaded_twice():
    """Each bundled library is mapped once: two copies of one library under
    different names are two images with separate state (a second
    libjulia-internal is an uninitialised runtime)."""
    import json
    import pathlib
    data = pathlib.Path(__file__).parent / "data" / "cases.json"
    r = subprocess.run([sys.executable, "-c", _IMAGES, str(data)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    images = json.loads(r.stdout.splitlines()[-1])
    assert any("libetcouple" in p for p in images)
    by_hash = {}
    for p, h in images.items():
        by_hash.setdefault(h, []).append(p)
    assert all(len(v) == 1 for v in by_hash.values()), [v for v in by_hash.values() if len(v) > 1]
