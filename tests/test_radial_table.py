"""Per-species-pair radial tables in r (`splinify.radial_table`): opt-in, an
approximation, not a restructuring.

The ACE table holds R_nl (Agnesi transform and envelope folded in) and the pair
radial; the PACE table holds the radial basis g_k (core repulsion stays
analytic).  Both are cubic B-splines on one uniform grid in r over
[r_min, max per-pair cutoff], sampled from the models' own exact code, and
masked to exact zeros beyond each pair's cutoff.  At the default 4000
intervals E, F and the virial agree with the analytic lean model to the bounds
below (measured, then given margin)."""
import dataclasses
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from conftest import pace_fixture
from test_lean import MODELS, _efv, _structure

from ace_jax.eval import load
from ace_jax.eval import splinify
from ace_jax.eval.model import lean
from ace_jax.eval.splinify import DEFAULT_RADIAL_TABLE, DEFAULT_TABLE_R_MIN, radial_table

ROOT = pathlib.Path(__file__).parent.parent
PACE = ROOT / "fixtures" / "pace"
ACE_NAMES = ["si_fitted", "si_ace_model", "sige_nofit", "Cantor_small"]
PACE_NAMES = ["gesi_sbessel", "si_chebexpcos", "sige_distance", "sige_zbl"]

# at 4000 intervals; the measured maxima over the fixtures here in the comments.
# si_ace_model sets F and V: its pair radial is a spline in x (jumps in the third
# derivative at its knots), and the virial sums r (x) each edge's error
E_REL = 1e-10         # |dE| / |E|            1.3e-12
F_REL = 1e-7          # max|dF| / max|F|      1.5e-8
V_REL = 1e-6          # max|dV| / max|V|      4.7e-7


def _lean(m):
    return lean(m, spline_tol=None)


def _close(a, b):
    (E0, F0, V0), (E1, F1, V1) = a, b
    dE, dF, dV = (abs(E1 - E0) / abs(E0), np.abs(F1 - F0).max() / np.abs(F0).max(),
                  np.abs(V1 - V0).max() / np.abs(V0).max())
    print(f"\n  dE/|E| {dE:.2e}  dF/max|F| {dF:.2e}  dV/max|V| {dV:.2e}")
    assert dE <= E_REL and dF <= F_REL and dV <= V_REL


@pytest.fixture(scope="module", params=ACE_NAMES)
def ace(request):
    m, meta, _ = load(str(MODELS[request.param]))
    m0 = _lean(m)
    return request.param, m0, lean(m, spline_tol=None, radial_table=True), meta, _structure(meta)


@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_ace_table_agrees_with_lean(ace, layout):
    name, m0, m1, meta, at = ace
    assert m1.rtab_coefs is not None and (m1.blk_rtab_coefs is not None) == m1.blk_compact
    _close(_efv(m0, meta, at, layout), _efv(m1, meta, at, layout))


def _pace(name):
    m, meta, _ = load(str(pace_fixture(PACE / f"{name}.yace")))
    return m, meta


@pytest.mark.parametrize("layout", ["sparse", "dense"])
@pytest.mark.parametrize("name", PACE_NAMES)
def test_pace_table_agrees(name, layout):
    m, meta = _pace(name)
    m1 = lean(m, radial_table=True)
    assert m1.rtab_coefs.shape == (*m.radparams.shape[:2], DEFAULT_RADIAL_TABLE + 3, m.nradbase)
    at = _structure(meta)
    _close(_efv(m, meta, at, layout), _efv(m1, meta, at, layout))


def test_info_and_defaults():
    m, meta, _ = load(str(MODELS["sige_nofit"]))
    m1, info = radial_table(_lean(m), return_info=True)
    assert m1.rtab_grid[2] == DEFAULT_RADIAL_TABLE + 1 and m1.rtab_grid[0] == DEFAULT_TABLE_R_MIN
    assert info["n_intervals"] == DEFAULT_RADIAL_TABLE and info["r_min"] == DEFAULT_TABLE_R_MIN
    assert info["r_max"] == pytest.approx(float(meta["rcut"]))
    assert 0 < info["max_rel_err"] < 1e-8 and 0 < info["max_rel_deriv_err"] < 1e-5
    m2 = lean(m, spline_tol=None, radial_table=500)
    assert m2.rtab_grid[2] == 501
    assert lean(m, spline_tol=None, radial_table=True).rtab_grid == m1.rtab_grid
    for bad in (0, -3, 2.5, "x"):
        with pytest.raises((ValueError, TypeError)):
            lean(m, radial_table=bad)


def test_off_is_unchanged():
    """radial_table=None (the default) adds no arrays and changes no result."""
    m, meta, _ = load(str(MODELS["Cantor_small"]))
    a, b = lean(m), lean(m, radial_table=None)
    assert b.rtab_coefs is None and b.blk_rtab_coefs is None and b.rtab_grid == ()
    assert jax.tree.structure(a) == jax.tree.structure(b)
    at = _structure(meta)
    for layout in ("sparse", "dense"):
        for x, y in zip(_efv(a, meta, at, layout), _efv(b, meta, at, layout)):
            np.testing.assert_array_equal(x, y)
    p, _ = _pace("gesi_sbessel")
    assert lean(p) is p and lean(p, radial_table=False) is p


def test_wrapper_refuses():
    from test_to_spline import Wrapper
    m, meta, _ = load(str(MODELS["sige_nofit"]))
    w = Wrapper(base=m)
    with pytest.raises(ValueError, match="basis"):
        lean(w, radial_table=True)
    assert lean(w).base is not None                        # off: unchanged behaviour


# ------------------------------------------------------------------ the cutoff
def _rij(r):
    r = jnp.asarray(r, jnp.float64)
    return jnp.stack([r, 0 * r, 0 * r], -1)


def test_ace_zero_beyond_cutoff():
    m, meta, _ = load(str(MODELS["Cantor_small"]))
    m1 = lean(m, spline_tol=None, radial_table=True)
    rc = float(meta["rcut"])
    r = np.array([rc, rc + 1e-9, rc + 0.3, rc + 1.0, rc + 5.0])
    z = jnp.zeros(len(r), jnp.int32)
    for zi in range(5):
        for zj in range(5):
            R, P = m1.radial(_rij(r), z + zi, z + zj)
            assert not np.any(np.asarray(R)) and not np.any(np.asarray(P))
            R, P = m1._blocked_radial(_rij(r), z + zi, z + zj)
            assert not np.any(np.asarray(R)) and not np.any(np.asarray(P))


def test_pace_per_pair_cutoff():
    """gesi_sbessel has bond cutoffs 4.2 and 5.0: a pair with rc = 4.2 gives exact
    zeros for 4.2 <= r < 5.0, inside the table's range."""
    m, meta = _pace("gesi_sbessel")
    m1 = lean(m, radial_table=True)
    rc = np.asarray(m.radparams[..., 1])
    i, j = np.unravel_index(np.argmin(rc), rc.shape)
    assert rc[i, j] < rc.max()
    r = np.linspace(rc[i, j], rc.max(), 7)
    z = jnp.zeros(len(r), jnp.int32)
    g, _ = m1.edge_basis_factors(_rij(r), z + int(i), z + int(j))
    assert not np.any(np.asarray(g))
    g, _ = m1.edge_basis_factors(_rij(r[:1] - 0.2), z[:1] + int(i), z[:1] + int(j))
    assert np.any(np.asarray(g))


def test_ace_radial_must_vanish_at_the_pair_cutoff():
    """The table masks at the pair envelope's cutoff, so R_nl must vanish there:
    a model where it does not is refused, not silently truncated."""
    m, meta, _ = load(str(MODELS["sige_nofit"]))
    pe = np.array(m.pair_envelope)
    pe[0, 1, 0] -= 1.0
    bad = dataclasses.replace(m, pair_envelope=jnp.asarray(pe))
    with pytest.raises(ValueError, match="vanish"):
        lean(bad, spline_tol=None, radial_table=True)


@pytest.mark.parametrize("path", ["ace", "pace"])
def test_skin_path_drops_skin_edges(path):
    """Skin-list edges between rcut and rcut + skin contribute nothing: skin=1
    agrees with skin=0 to roundoff with the table."""
    from ace_jax.calc.point import ACECalculator
    p = (str(MODELS["Cantor_small"]) if path == "ace"
         else str(pace_fixture(PACE / "gesi_sbessel.yace")))
    _, meta, _ = load(p)
    at = _structure(meta)
    res = []
    for skin in (0.0, 1.0):
        a = at.copy()
        a.calc = ACECalculator(p, layout="dense", skin=skin, radial_table=True)
        res.append((a.get_potential_energy(), a.get_forces()))
    assert abs(res[0][0] - res[1][0]) <= 1e-12 * abs(res[0][0])
    np.testing.assert_allclose(res[1][1], res[0][1], rtol=0, atol=1e-12)


# ------------------------------------------------------------------ calculator
@pytest.mark.parametrize("path", ["ace", "pace"])
def test_calculator(path):
    from ace_jax.calc.point import ACECalculator
    p = (str(MODELS["Cantor_small"]) if path == "ace"
         else str(pace_fixture(PACE / "gesi_sbessel.yace")))
    _, meta, _ = load(p)
    at = _structure(meta)
    out = {}
    for rt in (None, True):
        calc = ACECalculator(p, radial_table=rt)
        a = at.copy()
        a.calc = calc
        out[rt] = (a.get_potential_energy(), a.get_forces(), a.get_stress())
        if rt is None:
            assert calc.radial_table is None and calc.last_timing["radial_table"] is None
        else:
            assert calc.radial_table["n_intervals"] == DEFAULT_RADIAL_TABLE
            assert calc.last_timing["radial_table"] == DEFAULT_RADIAL_TABLE
            assert calc.eval_model.rtab_coefs is not None and calc.model.rtab_coefs is None
    (E0, F0, S0), (E1, F1, S1) = out[None], out[True]
    assert abs(E1 - E0) <= E_REL * abs(E0)
    assert np.abs(F1 - F0).max() <= F_REL * np.abs(F0).max()
    assert np.abs(S1 - S0).max() <= V_REL * np.abs(S0).max()


# ------------------------------------------------------------------ cache
@pytest.fixture
def builds(monkeypatch):
    """Counts the table fits `radial_table` actually runs (cache misses)."""
    splinify.clear_cache()
    n = {"fit": 0}
    real = splinify._rtab_fit

    def spy(*a, **k):
        n["fit"] += 1
        return real(*a, **k)

    monkeypatch.setattr(splinify, "_rtab_fit", spy)
    yield n
    splinify.clear_cache()


def test_cache_hits_on_readout_only_swap(builds):
    from ace_jax.calc.point import ACECalculator
    m, meta, _ = load(str(MODELS["sige_nofit"]))
    calc = ACECalculator(m, meta, radial_table=True)
    n0 = builds["fit"]
    assert n0 == 2                                         # full + compact
    first = calc.eval_model
    calc.model = dataclasses.replace(m, ctilde=2 * m.ctilde, E0=m.E0 + 1)
    assert builds["fit"] == n0
    np.testing.assert_array_equal(np.asarray(calc.eval_model.rtab_coefs),
                                  np.asarray(first.rtab_coefs))
    lean(m, spline_tol=None, radial_table=1000)            # n_intervals is part of the key
    assert builds["fit"] == n0 + 2
    calc.model = dataclasses.replace(m, Wpair=2 * m.Wpair)  # folded into the pair radial: misses
    assert builds["fit"] > n0 + 2


def test_pace_cache_hits_on_readout_only_swap(builds):
    m, _ = _pace("gesi_sbessel")
    lean(m, radial_table=True)
    assert builds["fit"] == 1
    lean(dataclasses.replace(m, crad=2 * m.crad, ctilde_complex=3 * m.ctilde_complex),
         radial_table=True)
    assert builds["fit"] == 1
    rp = np.array(m.radparams)
    rp[..., 0] *= 1.01                                     # lambda: the basis itself
    lean(dataclasses.replace(m, radparams=jnp.asarray(rp)), radial_table=True)
    assert builds["fit"] == 2


# ------------------------------------------------------------------ export
def test_export_records_the_table(tmp_path):
    from conftest import require_optional
    require_optional("lammps_jax")
    from test_export_lammps import LAMMPS_JAX_KEYS

    from ace_jax.export.lammps import export_lammps
    m, meta = _pace("gesi_sbessel")
    b = export_lammps(m, meta, tmp_path / "m.json", max_atoms=64, max_edges=64 * 64,
                      layout="sparse", radial_table=True)
    rt = b["ace_jax"]["radial_table"]
    assert rt["n_intervals"] == DEFAULT_RADIAL_TABLE and rt["r_min"] == DEFAULT_TABLE_R_MIN
    assert not (set(b["ace_jax"]) | set(rt)) & LAMMPS_JAX_KEYS
    assert b["ace_jax"]["lean"] is False                   # PACE: the table is not lean
    b = export_lammps(m, meta, tmp_path / "n.json", max_atoms=64, max_edges=64 * 64,
                      layout="sparse")
    assert b["ace_jax"]["radial_table"] is None
