"""REAL numerical parity: the Python ET/JuliaCall coupling bridge vs a
Julia/ACEpotentials reference (julia/coupling_reference.jl), bit-for-bit.

Skips cleanly without the `authoring` extra (core pip suite unaffected).  The
bridge is run in a SUBPROCESS -- juliacall's in-process init is unreliable under
pytest's import/capture machinery (documented in construct/coupling.py); a child
process is a genuine end-to-end run, not a mock.

PARITY IS ET-VERSION-SENSITIVE: the real-spherical-harmonic coupling convention
changed between EquivariantTensors 0.4.3 and 0.5.2.  The reference fixtures were
generated with the ET version ACEpotentials pins; juliapkg.json pins the bridge
to the SAME ET, so both sides agree to machine precision.  A version skew shows
up here as a large residual (that is the signal, not a flake).
"""
import json
import subprocess
import sys

import numpy as np
import pytest

from conftest import FIXTURE_DIR

pytest.importorskip("juliacall", reason="authoring extra (juliacall) not installed")

REFS = ["coupling_ref_SiGe_o2d6.npz", "coupling_ref_CrMnFe_o2d5.npz"]

_SCRIPT = r"""
import json, numpy as np, juliacall
from ace_jax.construct.spec import spec_from_reference
from ace_jax.construct.coupling import couple
mb, Rnl, Ylm, ref = spec_from_reference(REF_PATH)
A2B, aa, aspec = couple(mb, Rnl, Ylm)
print("RESULT", json.dumps({
    "shape_py": list(A2B.shape), "shape_ref": list(ref.shape),
    "max_abs_err": float(np.abs(A2B - ref).max()) if A2B.shape == ref.shape else None,
    "n_B": int(ref.shape[0]), "n_AA": int(ref.shape[1]), "nnz": int((ref != 0).sum()),
    "et_version": str(juliacall.Main.seval("using EquivariantTensors; string(pkgversion(EquivariantTensors))")),
}))
"""


@pytest.mark.parametrize("ref", REFS)
def test_coupling_parity(ref):
    path = FIXTURE_DIR / ref
    if not path.exists():
        pytest.skip(f"missing reference {ref} (regenerate with julia/coupling_reference.jl)")
    script = f"REF_PATH = {str(path)!r}\n" + _SCRIPT
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, f"bridge subprocess failed:\n{p.stderr[-3000:]}"
    r = json.loads([l for l in p.stdout.splitlines() if l.startswith("RESULT")][-1][len("RESULT "):])
    assert r["shape_py"] == r["shape_ref"], f"shape mismatch {r['shape_py']} vs {r['shape_ref']}"
    # bit-for-bit up to f64 roundoff; a version skew would make this ~O(1)
    assert r["max_abs_err"] < 1e-12, (
        f"coupling parity FAILED for {ref}: max|py-ref|={r['max_abs_err']:.3e} "
        f"(bridge ET {r['et_version']}); check the ET version matches the reference's.")
