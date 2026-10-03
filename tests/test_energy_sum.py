"""The total energy is a compensated sum of the site energies
(`edge_model.total_energy`): within ~1 ulp of `math.fsum`, returned in float64 for
float32 sites when x64 is on, with the derivative of a plain sum, so forces and
virials are bitwise unchanged.  docs/dev/energy-sum-results.md has the measurements."""
import math

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase.build import bulk
from ase.calculators.calculator import all_changes

import ace_jax.eval.edge_model as edge_model
from ace_jax.calc.point import ACECalculator
from ace_jax.eval.edge_model import compensated_sum, total_energy
from ace_jax.eval.nlist import dense_from_sparse, sparse_graph
from conftest import FIXTURE_DIR

M = str(FIXTURE_DIR / "si_fitted.npz")


def _ulps(got, exact):
    return abs(float(got) - exact) / np.spacing(abs(exact))


def _cases():
    rng = np.random.default_rng(0)
    return {
        "site_like": -163.17 + 0.05 * rng.standard_normal(200_001),        # odd n, E0-sized sites
        "cancelling": np.tile([1e8 + 0.1, -1e8], 50_000) + 1e-9 * rng.standard_normal(100_000),
        "big_small": np.concatenate([np.array([1e16, 1.0, -1e16] * 1000), rng.standard_normal(1000)]),
        "wide_range": rng.standard_normal(100_000) * 10.0 ** rng.integers(-10, 10, 100_000),
    }


@pytest.mark.parametrize("case", ["site_like", "cancelling", "big_small", "wide_range"])
def test_compensated_sum_matches_fsum(case):
    x = _cases()[case]
    exact = math.fsum(x)
    assert _ulps(jax.jit(compensated_sum)(jnp.asarray(x)), exact) <= 1
    if case in ("cancelling", "big_small"):         # where a plain sum is visibly wrong
        assert _ulps(jax.jit(jnp.sum)(jnp.asarray(x)), exact) > 1e3


def test_compensated_sum_shapes_and_order():
    rng = np.random.default_rng(1)
    for n in (0, 1, 2, 3, 5, 1024, 1025):
        x = rng.standard_normal(n) * 1e3
        assert float(compensated_sum(jnp.asarray(x))) == math.fsum(x)
    x = rng.standard_normal((7, 9))
    assert float(compensated_sum(jnp.asarray(x))) == math.fsum(x.ravel())
    # correctly rounded here, so independent of the order of the terms
    x = -163.17 + 0.05 * rng.standard_normal(60_000)
    f = jax.jit(compensated_sum)
    assert float(f(jnp.asarray(x))) == float(f(jnp.asarray(x[rng.permutation(len(x))])))


def test_gradient_is_that_of_a_plain_sum():
    x = jnp.asarray(_cases()["site_like"])
    assert bool(jnp.all(jax.grad(compensated_sum)(x) == 1.0))
    g_plain = jax.grad(lambda y: jnp.sum(jnp.sin(y) * y))(x)
    g_comp = jax.grad(lambda y: compensated_sum(jnp.sin(y) * y))(x)
    assert bool(jnp.all(g_plain == g_comp))
    assert float(jax.jvp(compensated_sum, (x,), (jnp.ones_like(x),))[1]) == x.shape[0]


def test_float32_sites_accumulate_in_float64():
    e = jnp.asarray(-163.17 + 0.05 * np.random.default_rng(2).standard_normal(100_000), jnp.float32)
    exact = math.fsum(np.asarray(e, np.float64))
    E = total_energy(e)
    assert E.dtype == jnp.float64 and _ulps(E, exact) <= 1
    assert abs(float(jnp.sum(e)) - exact) > 1e-3     # a float32 total: ulp(1.6e7) = 2 eV
    g = jax.grad(lambda y: total_energy(y))(e)
    assert g.dtype == jnp.float32 and bool(jnp.all(g == 1.0))


def _structure():
    at = bulk("Si", "diamond", a=5.45, cubic=True).repeat((3, 3, 3))
    at.rattle(0.05, seed=3)
    return at


def _dense_args(calc, at, dtype):
    g = sparse_graph(at.positions, at.cell.array, at.pbc, calc.cutoff)
    dg = dense_from_sparse(g, calc.cutoff)
    node_z = jnp.zeros(len(at), jnp.int32)
    idx = jnp.asarray(dg.idx, jnp.int32)
    mask = jnp.arange(idx.shape[1])[None, :] < jnp.asarray(dg.count)[:, None]
    zi = jnp.broadcast_to(node_z[:, None], idx.shape)
    return jnp.asarray(dg.rij, dtype), zi, node_z[idx], idx, mask, node_z


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_energy_compensated_forces_bitwise_unchanged(dtype, monkeypatch):
    """E within 1 ulp of fsum of the site energies; F and V exactly those of the
    plain jnp.sum the total used to be."""
    calc = ACECalculator(M, dtype=dtype)
    m, at = calc.eval_model, _structure()
    rij, zi, zj, idx, mask, node_z = _dense_args(calc, at, dtype)
    efv = lambda: jax.jit(lambda r: m.energy_forces_virial_dense(r, zi, zj, idx, mask, node_z))(rij)
    E, F, V = efv()
    monkeypatch.setattr(edge_model, "total_energy", jnp.sum)
    E0, F0, V0 = efv()
    assert bool(jnp.all(F == F0)) and bool(jnp.all(V == V0))
    sites = jax.jit(lambda r: m.site_energies_dense(r, zi, zj, mask, node_z))(rij)
    exact = math.fsum(np.asarray(sites, np.float64))
    assert E.dtype == jnp.float64
    if dtype == jnp.float64:
        assert _ulps(E, exact) <= 1
    else:   # float32 sites round differently in another compiled graph: compare loosely
        assert E0.dtype == jnp.float32 and abs(float(E) - exact) < 1e-3


def test_float32_calculator_energy_on_every_path():
    """float32 sites: the skin step, the fresh dense list and the sparse layout all
    return the float64 total (the skin step packs it as hi + lo in float32)."""
    at = _structure()
    Es = []
    for layout, skin in (("dense", 1.0), ("dense", 0.0), ("sparse", 0.0)):
        calc = ACECalculator(M, dtype=jnp.float32, layout=layout, skin=skin)
        calc.calculate(at, ["energy"], all_changes)
        assert calc.last_layout == layout
        Es.append(calc.results["energy"])
    assert abs(Es[0] - Es[1]) <= 2 * np.spacing(abs(Es[1]))
    assert abs(Es[0] - Es[2]) < 1e-3                 # float32 sites: per-site rounding only
    assert Es[0] != float(np.float32(Es[0]))         # not quantised to a float32 total
