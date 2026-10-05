"""Parity for the EquivariantTensors coupling shim (basis/coupling.py).

The bridge test needs a working `ace_jax_coupling` (the `basis` extra's
compiled library; `conftest.require_coupling_lib`) and runs in-process; the
spec port test is pure Python and always runs.  See basis/coupling.py and
docs/dev/coupling-etshim-spec.md.
"""
from conftest import require_coupling_lib

from ace_jax.basis.spec import rpe_admissible, ylm_spec


def test_rpe_admissible_port():
    """Port of ACEpotentials _rpe_filter_real(L=0)."""
    assert rpe_admissible([(1, 0, 0)])
    assert not rpe_admissible([(1, 1, 1)])                 # singleton l != 0
    assert rpe_admissible([(1, 1, 1), (2, 1, -1)])         # sum l even, m -> 0
    assert not rpe_admissible([(1, 1, 0), (2, 2, 0)])      # sum l = 3 odd


def test_bridge_wellformed():
    """The coupling library returns a coupling in the export layout."""
    require_coupling_lib()
    from ace_jax.basis.coupling import couple
    cpl = couple([[(1, 0)], [(1, 1), (1, 1)]], [(1, 0), (1, 1), (2, 0)], ylm_spec(1))
    A2B, aa_sig, aspec = cpl.A2B, cpl.aa_sig, cpl.aspec
    assert A2B.shape[1] == len(aa_sig)                     # (n_B, n_AA): one aa_sig per column
    assert A2B.shape[0] >= 1 and int((A2B != 0).sum()) >= 1
    assert all(isinstance(s, tuple) and all(len(t) == 3 for t in s) for s in aa_sig)
    assert all(list(s) == sorted(s) for s in aa_sig)
    assert all(0 <= l <= 1 for s in aa_sig for (n, l, m) in s)
    assert len(set(aa_sig)) == len(aa_sig)                 # each AA column is a distinct function
    assert all(0 <= r < 3 and 0 <= y < 4 for r, y in aspec)


def test_couple_columns_share_aa_specs_order():
    """A2B column j multiplies the j-th AA product of the evaluation order (the
    concatenated aa_specs rows), so aa_sig[j] must be that product's (n, l, m)
    signature.  build_spec interleaves body orders, which ET's SparseSymmProd
    regroups by length while the A2B columns stay in 𝔸spec order."""
    require_coupling_lib()
    from ace_jax.basis.coupling import couple
    from ace_jax.basis.spec import build_spec
    mb, Rnl, Ylm = build_spec(1, 3, 6)
    assert [len(b) for b in mb] != sorted(len(b) for b in mb)     # the interleaved case
    cpl = couple(mb, Rnl, Ylm)
    rows = [r for g in cpl.aa_specs for r in g]
    ev = [tuple(sorted((*Rnl[cpl.aspec[a][0]], Ylm[cpl.aspec[a][1]][1]) for a in r)) for r in rows]
    assert ev == [tuple(sorted(s)) for s in cpl.aa_sig]
