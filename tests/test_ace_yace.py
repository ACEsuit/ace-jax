"""A linear ACEModel written as a `.yace` (export/yace.py) and read back (pace_io, radbasename Tabulated)
evaluates as the source model: the radial tables are its own values and derivatives, the products a
re-expression of its coefficients.  ML-PACE parity needs its tabulated-radial addition, so it is not
tested here (docs/dev/ml-pace-tabulated-radials.md)."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from ase.build import bulk  # noqa: E402

import ace_jax as aj  # noqa: E402
from ace_jax.calc.point import ACECalculator  # noqa: E402
from ace_jax.eval import write_yace  # noqa: E402
from ace_jax.eval.pace_io import load_tree  # noqa: E402
from conftest import FIXTURE_DIR  # noqa: E402


def _cell(species):
    at = bulk("Si", "diamond", 5.43, cubic=True).repeat(2)
    at.rattle(0.1, seed=1)
    at.numbers[:] = species[0]
    for k, z in enumerate(species[1:], 1):
        at.numbers[k::len(species)] = z
    return at


def _efs(calc, at):
    at = at.copy()
    at.calc = calc
    return at.get_potential_energy(), at.get_forces(), at.get_stress()


# (fixture, species in the test cell, table spacing): the default 1e-3 A on Si; coarser on the
# multi-species fixtures, whose NZ^2 bond tables are otherwise tens of MB
CASES = [("si_ace_model.npz", (14,), 1e-3), ("sige_nofit.npz", (14, 32), 5e-3),
         ("ace_cantor5_small.npz", (24, 25, 26, 27, 28), 1e-2)]


@pytest.mark.parametrize("name,species,delta", CASES, ids=[c[0].split(".")[0] for c in CASES])
def test_yace_export_evaluates_as_the_source(tmp_path, name, species, delta):
    src = FIXTURE_DIR / name
    model, _, _ = aj.load(src)
    info = write_yace(model, tmp_path / "m.yace", delta=delta)
    assert info["max_imag_rel"] < 1e-12                     # invariants: real C-tilde coefficients
    at = _cell(species)
    E0, F0, S0 = _efs(ACECalculator(src, skin=0), at)
    E1, F1, S1 = _efs(ACECalculator(tmp_path / "m.yace", skin=0), at)
    # cubic Hermite in r between table nodes: O(h^4) values, O(h^3) derivatives
    ftol = {1e-3: 3e-8, 5e-3: 3e-6, 1e-2: 3e-5}[delta]
    assert abs(E1 - E0) / len(at) < 10 * ftol
    assert np.abs(F1 - F0).max() < ftol * max(1.0, np.abs(F0).max())
    assert np.abs(S1 - S0).max() < ftol * max(1.0, np.abs(S0).max())


def test_yace_export_writes_half_basis_and_trimmed_shape(tmp_path):
    model, _, _ = aj.load(FIXTURE_DIR / "si_ace_model.npz")
    write_yace(model, tmp_path / "m.yace")
    t = load_tree(tmp_path / "m.yace")
    b = t["bonds"][(0, 0)]
    funcs = [f for f in t["functions"][0] if f["rank"] > 1]
    # lmax / nradmax as the products read them, not the model's
    assert b["lmax"] == max(max(f["ls"]) for f in funcs) < model.lmax
    assert b["nradmax"] == max(max(f["ns"]) for f in funcs)
    assert b["radbasename"] == "Tabulated" and b["radtable"]["nbins"] == round(b["rcut"] / t["deltaSplineBins"])
    for f in funcs:                                          # half basis: no m-tuple beside its negation
        r = f["rank"]
        keys = {tuple(sorted(zip(f["mus"], f["ns"], f["ls"], f["ms_combs"][k * r:(k + 1) * r])))
                for k in range(f["num_ms_combs"])}
        negs = {tuple(sorted((mu, n, l, -m) for mu, n, l, m in k)) for k in keys}
        assert all(neg == k or neg not in keys for k, neg in
                   ((k, tuple(sorted((mu, n, l, -m) for mu, n, l, m in k))) for k in keys)), negs


def test_yace_tabulated_rewrite_round_trips(tmp_path):
    """A tabulated PACEModel written back (write_yace with its spec) keeps its tables and evaluates the same."""
    model, _, _ = aj.load(FIXTURE_DIR / "si_ace_model.npz")
    write_yace(model, tmp_path / "a.yace")
    pm, _, spec = aj.load(tmp_path / "a.yace")
    write_yace(pm, tmp_path / "b.yace", spec=spec)
    assert load_tree(tmp_path / "b.yace") == load_tree(tmp_path / "a.yace")
    at = _cell((14,))     # two calculators may calibrate gather / matmul differently: roundoff, not bits
    Ea, Fa, _ = _efs(ACECalculator(tmp_path / "a.yace", skin=0), at)
    Eb, Fb, _ = _efs(ACECalculator(tmp_path / "b.yace", skin=0), at)
    assert abs(Ea - Eb) < 1e-12 * abs(Ea) and np.abs(Fa - Fb).max() < 1e-12 * np.abs(Fa).max()


def test_write_yace_arguments(tmp_path):
    model, _, _ = aj.load(FIXTURE_DIR / "si_ace_model.npz")
    with pytest.raises(TypeError, match="no spec"):
        write_yace(model, tmp_path / "m.yace", spec=object())
    write_yace(model, tmp_path / "m.yace", delta=1e-2)
    pm, _, spec = aj.load(tmp_path / "m.yace")
    with pytest.raises(TypeError, match="needs the spec"):
        write_yace(pm, tmp_path / "p.yace")
    with pytest.warns(DeprecationWarning, match="deprecated"):
        write_yace(pm, spec, tmp_path / "old-order.yace")
    assert (tmp_path / "old-order.yace").exists()
    with pytest.raises(ValueError, match="whole number"):
        write_yace(model, tmp_path / "x.yace", delta=0.003)
    from ace_jax.eval.model import lean
    with pytest.raises(Exception, match="lean|full"):
        write_yace(lean(model), tmp_path / "lean.yace")


def test_yace_export_summed_pools_neighbour_species(tmp_path):
    """summed=True keeps ACE's pooled A: every factor is element 0, one rank-1 g_1, each bond's tables hold
    every column, and the tree flags summed_neighbour_A (an ML-PACE addition this package's reader refuses);
    the default expands each factor over the live neighbour species instead."""
    model, _, _ = aj.load(FIXTURE_DIR / "sige_nofit.npz")
    info_s = write_yace(model, tmp_path / "s.yace", delta=5e-3, summed=True)
    info_e = write_yace(model, tmp_path / "e.yace", delta=5e-3)
    t = load_tree(tmp_path / "s.yace")
    assert t["summed_neighbour_A"] is True and "summed_neighbour_A" not in load_tree(tmp_path / "e.yace")
    for fl in t["functions"].values():
        assert all(mu == 0 for f in fl for mu in f["mus"])
        assert sum(f["rank"] == 1 for f in fl) == 1
    ns = {(b["radtable"]["rnl"][k]["n"], b["radtable"]["rnl"][k]["l"])
          for b in t["bonds"].values() for k in range(len(b["radtable"]["rnl"]))}
    for b in t["bonds"].values():                            # the same (n, l) set on every bond
        assert {(e["n"], e["l"]) for e in b["radtable"]["rnl"]} == ns
    assert info_s["terms"] < info_e["terms"]
    with pytest.raises(NotImplementedError, match="summed_neighbour_A"):
        aj.load(tmp_path / "s.yace")
