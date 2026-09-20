"""Parity for the EquivariantTensors coupling shim (construct/coupling.py).

Skips cleanly when the optional `authoring` extra (juliacall) is absent, so the
197-test core pip suite is unaffected.

The ET bridge is exercised in a SUBPROCESS (a clean interpreter): juliacall's
in-process initialisation is unreliable under pytest's import/capture machinery
(it reports "Package EquivariantTensors not found in current path" even though
the same call succeeds in a plain `python -c`), so we run the real bridge in a
child process and check its output -- a genuine end-to-end validation, not a
mock.  See construct/coupling.py and docs/coupling-etshim-spec.md.
"""
import json
import subprocess
import sys

import pytest

from conftest import FIXTURE_DIR

pytest.importorskip("juliacall", reason="authoring extra (juliacall) not installed")

from ace_jax.construct.spec import rpe_admissible, ylm_spec, spec_from_export  # noqa: E402


def test_rpe_admissible_port():
    """Port of ACEpotentials _rpe_filter_real(L=0)."""
    assert rpe_admissible([(1, 0, 0)])
    assert not rpe_admissible([(1, 1, 1)])                 # singleton l != 0
    assert rpe_admissible([(1, 1, 1), (2, 1, -1)])         # sum l even, m -> 0
    assert not rpe_admissible([(1, 1, 0), (2, 2, 0)])      # sum l = 3 odd


_BRIDGE = r"""
import json, numpy as np, juliacall
from ace_jax.construct.spec import ylm_spec
from ace_jax.construct.coupling import couple
A2B, aa_sig, aspec = couple([[(1,0)], [(1,1),(1,1)]], [(1,0),(1,1),(2,0)], ylm_spec(1))
print("RESULT", json.dumps({
    "A2B_shape": list(A2B.shape),
    "A2B_nnz": int((A2B != 0).sum()),
    "n_AA": len(aa_sig),
    "sigs_ok": all(isinstance(s, tuple) and all(len(t) == 3 for t in s) for s in aa_sig),
    "sigs_sorted": all(list(s) == sorted(s) for s in aa_sig),
    "l_in_range": all(0 <= l <= 1 for s in aa_sig for (n, l, m) in s),
    "sigs_unique": len(set(aa_sig)) == len(aa_sig),   # each AA column is a distinct function
    "n_aspec": len(aspec),
    "aspec_ok": all(0 <= r < 3 and 0 <= y < 4 for r, y in aspec),
}))
"""


def test_bridge_wellformed_subprocess():
    """The in-process ET call returns a coupling in the export layout (run in a
    clean child process to avoid the pytest/juliacall init quirk)."""
    p = subprocess.run([sys.executable, "-c", _BRIDGE], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, f"bridge subprocess failed:\n{p.stderr[-2000:]}"
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")][-1]
    r = json.loads(line[len("RESULT "):])
    assert r["A2B_shape"][1] == r["n_AA"]                  # (n_B, n_AA): one aa_sig per column
    assert r["A2B_shape"][0] >= 1 and r["A2B_nnz"] >= 1
    assert r["sigs_ok"] and r["sigs_sorted"] and r["l_in_range"] and r["sigs_unique"]
    assert r["aspec_ok"]
