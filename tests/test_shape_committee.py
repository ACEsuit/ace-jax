"""The served shape as an r-output linear ACE (docs/dev/force-uq-vs-calm.md, Step 5).

v(x) = sum_a |R^T D^-1 phi_a|^2 and R^T D^-1 phi_a is the force on the atom of a linear ACE model with the
coefficient matrix C = D^-1 R (L x r): `committee_shape` evaluates those r forces from one basis evaluation,
contracting C into dB/dA per node before the edge derivative, so neither the force design rows (N, 3, L)
nor the edge Jacobian (E, D, 3) exist.  It must equal `atom_shape` over the rows to roundoff."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from conftest import FIXTURE_DIR  # noqa: E402

CASES = {"si": ("si_fitted.npz", "Si", "diamond", 5.43, (2, 2, 2)),
         "cantor5": ("ace_cantor5_small.npz", "Ni", "fcc", 3.60, (3, 3, 3))}


def _setup(name, seed=0):
    from ase.build import bulk

    from ace_jax.eval import load
    from ace_jax.fit.data import Config, build_dataset
    from ace_jax.fit.inducing import GPConfig
    path = FIXTURE_DIR / CASES[name][0]
    if not path.exists():
        pytest.skip(f"missing {path.name}")
    model, meta, z = load(path)
    _, el, lat, a, rep = CASES[name]
    at = bulk(el, lat, a=a, cubic=True).repeat(rep)
    rng = np.random.default_rng(seed)
    at.set_chemical_symbols(rng.choice(meta["elements"], len(at)))
    at.rattle(0.08, seed=seed + 1)
    c = Config(at.positions, at.numbers, at.cell.array, at.pbc, None, None, None, 1.0, 1.0, 1.0)
    ds = build_dataset([c], meta, np.zeros(len(meta["elements"])), 1)
    b = jax.tree.map(lambda x: x[0], ds)
    g = GPConfig(r0=2.4, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                 NZ=len(meta["elements"]), C=1)
    return model, g, b


def _rows_shape(model, g, b, R, dinv):
    from ace_jax.fit.jackknife import atom_shape
    from ace_jax.fit.rows import linear_rows
    F = np.asarray(linear_rows(model, g, b)[0].F)
    return atom_shape(R, dinv, F)


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("chunk", [None, 7])
def test_committee_shape_equals_rows_shape(name, chunk):
    from ace_jax.fit.jackknife import committee_shape
    model, g, b = _setup(name)
    L = g.len_basis
    rng = np.random.default_rng(1)
    R = rng.normal(size=(L, 5))
    dinv = rng.uniform(0.5, 2.0, L)
    V = committee_shape(model, g, b, R, dinv, node_chunk=chunk)
    Vr = _rows_shape(model, g, b, R, dinv)
    live = np.asarray(b.node_mask)
    scale = np.abs(Vr).max()
    assert scale > 0
    np.testing.assert_allclose(V[live], Vr[live], rtol=1e-11, atol=1e-12 * scale)
    assert np.all(V[~live] == 0)


def test_committee_ignores_trailing_e0_columns():
    """A joint-E0 posterior's R has NZ E0 rows after the readout; force rows are zero there."""
    from ace_jax.fit.jackknife import committee_shape
    model, g, b = _setup("si")
    rng = np.random.default_rng(2)
    R = rng.normal(size=(g.len_basis, 4))
    dinv = rng.uniform(0.5, 2.0, g.len_basis)
    V = committee_shape(model, g, b, R, dinv)
    Re = np.r_[R, rng.normal(size=(g.NZ, 4))]
    de = np.r_[dinv, np.ones(g.NZ)]
    np.testing.assert_array_equal(committee_shape(model, g, b, Re, de), V)


def test_truncate_shape_factor():
    from ace_jax.fit.jackknife import truncate_shape_factor
    rng = np.random.default_rng(3)
    R = rng.normal(size=(40, 12)) @ np.diag(2.0 ** -np.arange(12))
    Rt = truncate_shape_factor(R, 1.0)
    np.testing.assert_allclose(Rt @ Rt.T, R @ R.T, atol=1e-12)
    s = np.linalg.svd(R, compute_uv=False)
    for tau in (0.5, 0.9, 0.99):
        Rt = truncate_shape_factor(R, tau)
        r = Rt.shape[1]
        e = np.cumsum(s ** 2) / np.sum(s ** 2)
        assert e[r - 1] >= tau - 1e-12 and (r == 1 or e[r - 2] < tau)
        np.testing.assert_allclose(np.linalg.svd(Rt, compute_uv=False), s[:r], rtol=1e-12)
    assert truncate_shape_factor(R, 3).shape == (40, 3)            # an int is a rank
