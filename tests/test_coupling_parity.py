"""Numerical parity of the Python coupling path vs a Julia/ACEpotentials reference
(julia/coupling_reference.jl).

All tests skip cleanly without the `authoring` extra (the core pip suite stays
Julia-free) and run the bridge in a SUBPROCESS (juliacall's in-process init is
unreliable under pytest -- documented in construct/coupling.py).

Levels (each self-configures NZ/order/totaldegree/wL from the fixture metadata):

  * bridge parity      -- feed the ORACLE's own mb_spec through the ET/JuliaCall
                          bridge and compare to the reference coupling.
  * END-TO-END parity  -- generate mb_spec FROM SCRATCH in Python (construct.spec
                          build_spec, reproducing ACEpotentials' TotalDegree/wL
                          selection), feed it through the bridge, and compare.

THE COMPARISON (both levels).  A2B is block-diagonal in nnll: each B function has
a definite nnll and couples only the AA functions (m-realisations) of that same
nnll.  We group basis ROWS by nnll and align AA COLUMNS by their (n,l,m)
signature (globally unique), then per nnll block:
  * multiplicity 1  -> the single row must match the reference up to a sign
                       (permutation + per-row sign), asserted exactly.
  * multiplicity >1 -> at order >= 4 a degenerate block's symmetrisation basis is
                       unique only up to a within-block ORTHOGONAL rotation (a
                       different mb enumeration order rotates it), so exact A2B
                       parity is a SUBSPACE match, not permutation+sign.  We
                       compare the orthogonal projector onto each block's row
                       space (P = Q Qᵀ, Q an orthonormal basis of the row space
                       from a QR) -- invariant to permutation, sign AND rotation.
All residuals must be < 1e-12.

Column identities come from the tensor's meta `𝔸spec` (the spec returned WITH the
symmetrisation matrix), which is the true A2B column order -- see
construct/coupling.py.  ET's AA column ENUMERATION is deterministic in the mb
order; only the row (symmetrisation) basis of a degenerate block rotates between
runs, which is exactly what the projector invariant absorbs.

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

# Fixtures span orders 2, 3 (production: the Cantor model is order 3) and 4.
# Order 2/3 nnll blocks are all multiplicity 1; the order-4 SiGe fixture
# (wL=0.5, totaldegree 5) has degenerate blocks (252 B functions, 250 nnll).
# NZ/order/totaldegree/wL are read from each fixture, so the list is just names.
FIXTURES = [
    "coupling_ref_SiGe_o2d6.npz",     # order 2, 2 species
    "coupling_ref_CrMnFe_o2d5.npz",   # order 2, 3 species
    "coupling_ref_CrMnFe_o3d6.npz",   # order 3, 3 species  (all mult-1)
    "coupling_ref_SiGe_o4d5w05.npz",  # order 4, 2 species  (DEGENERATE blocks)
]


def _run(script):
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=1200)
    assert p.returncode == 0, f"bridge subprocess failed:\n{p.stderr[-3000:]}"
    return json.loads([l for l in p.stdout.splitlines() if l.startswith("RESULT")][-1][len("RESULT "):])


# Shared subprocess body.  MODE selects the mb_spec source:
#   "oracle" -> reconstruct the reference's own mb_spec (isolates the ET bridge)
#   "build"  -> generate mb_spec FROM SCRATCH with build_spec (end-to-end); also
#               asserts the generated mb SET equals the oracle's exactly.
_PARITY = r"""
import json, numpy as np, juliacall
from ace_jax.construct.spec import build_spec, spec_from_reference, _unflat
from ace_jax.construct.coupling import couple, subspace_residual

z = np.load(REF)
NZ, ORDER, TD, WL = int(z["n_elements"]), int(z["order"]), int(z["totaldegree"]), float(z["wL"])
A2B_ref = np.zeros(tuple(int(v) for v in z["A2B_shape"]))
A2B_ref[z["A2B_rows"], z["A2B_cols"]] = z["A2B_vals"]
aa_sig_ref = [tuple(sorted(tuple(int(x) for x in t) for t in tt))
              for tt in _unflat(z["aasig_flat"], z["aasig_len"], 3)]

mset_ok = None
if MODE == "oracle":
    mb, Rnl, Ylm, _ = spec_from_reference(REF)
else:
    mb, Rnl, Ylm = build_spec(NZ, ORDER, TD, wL=WL)
    norm = lambda body: tuple(sorted((int(n), int(l)) for n, l in body))
    oracle_mb = {norm(b) for b in _unflat(z["mb_flat"], z["mb_len"], 2)}
    build_mb = {norm(b) for b in mb}
    mset_ok = bool(build_mb == oracle_mb)

A2B_py, aa_sig_py, _aspec = couple(mb, Rnl, Ylm)
aa_sig_py = [tuple(sorted(s)) for s in aa_sig_py]

nl_of = lambda s: tuple(sorted((n, l) for (n, l, m) in s))
out = {"mode": MODE, "shape_ok": list(A2B_py.shape) == list(A2B_ref.shape), "mset_ok": mset_ok}
if out["shape_ok"] and (mset_ok is None or mset_ok):
    # AA column bijection py -> ref by (n,l,m) signature (globally unique).
    ref_col = {s: j for j, s in enumerate(aa_sig_ref)}
    out["ref_cols_unique"] = (len(ref_col) == len(aa_sig_ref))
    out["py_cols_unique"] = (len(set(aa_sig_py)) == len(aa_sig_py))
    if out["ref_cols_unique"] and out["py_cols_unique"]:
        perm = np.array([ref_col[s] for s in aa_sig_py])       # py col j -> ref col perm[j]
        A2B_py_r = np.zeros_like(A2B_ref)
        A2B_py_r[:, perm] = A2B_py                              # py columns in ref order

        def rows_by_nnll(A2B):
            d = {}
            for b in range(A2B.shape[0]):
                nz = np.flatnonzero(np.abs(A2B[b]) > 1e-12)
                d.setdefault(nl_of(aa_sig_ref[nz[0]]), []).append(b)
            return d
        py_rows, rf_rows = rows_by_nnll(A2B_py_r), rows_by_nnll(A2B_ref)
        cols_by_nnll = {}
        for j, s in enumerate(aa_sig_ref):
            cols_by_nnll.setdefault(nl_of(s), []).append(j)

        out["blocks_ok"] = (set(py_rows) == set(rf_rows))
        worst, worst_blk, n_mult1, n_degen = 0.0, None, 0, 0
        mult_mismatch = None
        if out["blocks_ok"]:
            for nnll in rf_rows:
                Rp, Rr, C = py_rows[nnll], rf_rows[nnll], cols_by_nnll[nnll]
                if len(Rp) != len(Rr):
                    mult_mismatch = [list(map(list, nnll)), len(Rp), len(Rr)]
                    break
                Mp, Mr = A2B_py_r[np.ix_(Rp, C)], A2B_ref[np.ix_(Rr, C)]
                if len(Rr) == 1:                                # multiplicity 1: perm + sign
                    n_mult1 += 1
                    res = float(min(np.abs(Mp - Mr).max(), np.abs(Mp + Mr).max()))
                else:                                            # degenerate: subspace (projector)
                    n_degen += 1
                    res = subspace_residual(Mp, Mr)
                if res > worst:
                    worst, worst_blk = res, [list(map(list, nnll)), len(Rr), res]
        out.update(max_residual=worst, worst_block=worst_blk, n_mult1=n_mult1,
                   n_degen=n_degen, mult_mismatch=mult_mismatch)
print("RESULT", json.dumps(out))
"""


def _parity(ref, mode):
    path = FIXTURE_DIR / ref
    if not path.exists():
        pytest.skip(f"missing reference {ref}")
    return _run(f"REF = {str(path)!r}; MODE = {mode!r}\n" + _PARITY)


@pytest.mark.parametrize("ref", FIXTURES)
def test_coupling_bridge_parity(ref):
    """Oracle mb_spec -> ET bridge -> reference coupling (isolates the bridge)."""
    r = _parity(ref, "oracle")
    assert r["shape_ok"], f"shape mismatch: {r}"
    assert r.get("ref_cols_unique") and r.get("py_cols_unique"), f"non-unique column sigs: {r}"
    assert r.get("blocks_ok"), f"nnll block sets differ: {r}"
    assert r.get("mult_mismatch") is None, f"block multiplicity mismatch: {r}"
    assert r["max_residual"] < 1e-12, f"bridge parity failed for {ref}: {r}"


@pytest.mark.parametrize("ref", FIXTURES)
def test_coupling_end_to_end_parity(ref):
    """Python-generated mb_spec (build_spec) -> ET bridge -> reference coupling.

    Asserts the generated mb SET matches ACEpotentials exactly, then the per-nnll
    coupling parity (perm+sign for mult-1 blocks, row-space subspace match for
    degenerate blocks at order >= 4)."""
    r = _parity(ref, "build")
    assert r["mset_ok"], f"build_spec mb SET != oracle mb SET for {ref}: {r}"
    assert r["shape_ok"], f"shape mismatch: {r}"
    assert r.get("ref_cols_unique") and r.get("py_cols_unique"), f"non-unique column sigs: {r}"
    assert r.get("blocks_ok"), f"nnll block sets differ: {r}"
    assert r.get("mult_mismatch") is None, f"block multiplicity mismatch: {r}"
    assert r["max_residual"] < 1e-12, f"end-to-end parity failed for {ref}: {r}"
