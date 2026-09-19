"""Numerical parity of the Python coupling path vs a Julia/ACEpotentials reference
(julia/coupling_reference.jl), bit-for-bit.

Two levels, both skip cleanly without the `authoring` extra (core pip suite stays
Julia-free) and run the bridge in a SUBPROCESS (juliacall's in-process init is
unreliable under pytest -- documented in construct/coupling.py):

  * bridge parity      -- feed the ORACLE's mb_spec through the ET/JuliaCall
                          bridge; A2B must equal the reference exactly.
  * END-TO-END parity  -- generate mb_spec FROM SCRATCH in Python (construct.spec
                          build_spec, reproducing ACEpotentials' TotalDegree/wL
                          selection), feed it through the bridge, align to the
                          reference by invariant signatures (row = per-B nnll,
                          col = per-AA (n,l,m); unique -> a pure permutation at
                          these orders) with a per-row sign, and assert exact.

PARITY IS ET-VERSION-SENSITIVE (the real-harmonic coupling convention changed
0.4.3 -> 0.5.2); juliapkg.json pins the bridge to the ET the reference was built
with, and julia-parity.yml keeps them in lockstep in CI.
"""
import json
import subprocess
import sys

import pytest

from conftest import FIXTURE_DIR

pytest.importorskip("juliacall", reason="authoring extra (juliacall) not installed")

# (fixture, NZ, order, totaldegree)
SHAPES = [("coupling_ref_SiGe_o2d6.npz", 2, 2, 6), ("coupling_ref_CrMnFe_o2d5.npz", 3, 2, 5)]


def _run(script):
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, f"bridge subprocess failed:\n{p.stderr[-3000:]}"
    return json.loads([l for l in p.stdout.splitlines() if l.startswith("RESULT")][-1][len("RESULT "):])


_BRIDGE = r"""
import json, numpy as np, juliacall
from ace_jax.construct.spec import spec_from_reference
from ace_jax.construct.coupling import couple
mb, Rnl, Ylm, ref = spec_from_reference(REF)
A2B, aa, aspec = couple(mb, Rnl, Ylm)
print("RESULT", json.dumps({"shape_ok": list(A2B.shape) == list(ref.shape),
      "max_abs_err": float(np.abs(A2B - ref).max()) if A2B.shape == ref.shape else None}))
"""

_E2E = r"""
import json, numpy as np, juliacall
from ace_jax.construct.spec import build_spec, spec_from_reference, _unflat
from ace_jax.construct.coupling import couple
z = np.load(REF); mb, Rnl, Ylm = build_spec(NZ, ORDER, TD)   # from-scratch Python spec
A2B, aa, aspec = couple(mb, Rnl, Ylm)
ref = np.zeros(tuple(map(int, z["A2B_shape"]))); ref[z["A2B_rows"], z["A2B_cols"]] = z["A2B_vals"]
def aasig(a_idx):
    return tuple(sorted((Rnl[aspec[a][0]][0], Rnl[aspec[a][0]][1], Ylm[aspec[a][1]][1]) for a in a_idx))
aasig_py = [aasig([int(x) for x in col]) for s in aa for col in s]
nnll_py = [tuple(sorted((n, l) for (n, l, m) in aasig_py[int(np.flatnonzero(A2B[b])[0])]))
           for b in range(A2B.shape[0])]
nnll_ref = [tuple(sorted(b)) for b in _unflat(z["nnll_flat"], z["nnll_len"], 2)]
aasig_ref = [tuple(sorted(t)) for t in _unflat(z["aasig_flat"], z["aasig_len"], 3)]
uniq = len(set(nnll_ref)) == len(nnll_ref) and len(set(aasig_ref)) == len(aasig_ref)
out = {"shape_ok": list(A2B.shape) == list(ref.shape), "unique": bool(uniq), "max_abs_err": None}
if A2B.shape == ref.shape and uniq:
    refcol = {}; [refcol.setdefault(s, []).append(j) for j, s in enumerate(aasig_ref)]
    pycol = {}; [pycol.setdefault(s, []).append(j) for j, s in enumerate(aasig_py)]
    colmap = np.empty(len(aasig_py), int)
    for s, js in pycol.items():
        for a, b in zip(js, refcol[s]): colmap[a] = b
    refrow = {s: i for i, s in enumerate(nnll_ref)}
    P = np.zeros_like(ref)
    for i in range(A2B.shape[0]): P[refrow[nnll_py[i]], colmap] = A2B[i]
    for r in range(ref.shape[0]):
        if np.dot(P[r], ref[r]) < 0: P[r] = -P[r]
    out["max_abs_err"] = float(np.abs(P - ref).max())
print("RESULT", json.dumps(out))
"""


@pytest.mark.parametrize("ref,NZ,order,td", SHAPES)
def test_coupling_bridge_parity(ref, NZ, order, td):
    path = FIXTURE_DIR / ref
    if not path.exists():
        pytest.skip(f"missing reference {ref}")
    r = _run(f"REF = {str(path)!r}\n" + _BRIDGE)
    assert r["shape_ok"] and r["max_abs_err"] < 1e-12, r


@pytest.mark.parametrize("ref,NZ,order,td", SHAPES)
def test_coupling_end_to_end_parity(ref, NZ, order, td):
    """Python-generated mb_spec -> ET -> == ACEpotentials reference (bit-for-bit,
    up to the resolved signature permutation + per-row sign)."""
    path = FIXTURE_DIR / ref
    if not path.exists():
        pytest.skip(f"missing reference {ref}")
    r = _run(f"REF = {str(path)!r}; NZ = {NZ}; ORDER = {order}; TD = {td}\n" + _E2E)
    assert r["shape_ok"], f"shape mismatch: {r}"
    assert r["unique"], f"signatures not unique (degeneracy) -- alignment needs subspace match: {r}"
    assert r["max_abs_err"] < 1e-12, f"end-to-end parity failed for {ref}: {r}"
