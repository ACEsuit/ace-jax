"""Parity for the EquivariantTensors coupling shim (construct/coupling.py).

The bridge test needs a working `ace_jax_coupling` (the `authoring` extra's
compiled library; `conftest.require_coupling_lib`) and runs in-process; the
spec port test is pure Python and always runs.  See construct/coupling.py and
docs/coupling-etshim-spec.md.
"""
from conftest import require_coupling_lib

from ace_jax.construct.spec import rpe_admissible, ylm_spec


def test_rpe_admissible_port():
    """Port of ACEpotentials _rpe_filter_real(L=0)."""
    assert rpe_admissible([(1, 0, 0)])
    assert not rpe_admissible([(1, 1, 1)])                 # singleton l != 0
    assert rpe_admissible([(1, 1, 1), (2, 1, -1)])         # sum l even, m -> 0
    assert not rpe_admissible([(1, 1, 0), (2, 2, 0)])      # sum l = 3 odd


def test_bridge_wellformed():
    """The coupling library returns a coupling in the export layout."""
    require_coupling_lib()
    from ace_jax.construct.coupling import couple
    cpl = couple([[(1, 0)], [(1, 1), (1, 1)]], [(1, 0), (1, 1), (2, 0)], ylm_spec(1))
    A2B, aa_sig, aspec = cpl.A2B, cpl.aa_sig, cpl.aspec
    assert A2B.shape[1] == len(aa_sig)                     # (n_B, n_AA): one aa_sig per column
    assert A2B.shape[0] >= 1 and int((A2B != 0).sum()) >= 1
    assert all(isinstance(s, tuple) and all(len(t) == 3 for t in s) for s in aa_sig)
    assert all(list(s) == sorted(s) for s in aa_sig)
    assert all(0 <= l <= 1 for s in aa_sig for (n, l, m) in s)
    assert len(set(aa_sig)) == len(aa_sig)                 # each AA column is a distinct function
    assert all(0 <= r < 3 and 0 <= y < 4 for r, y in aspec)
