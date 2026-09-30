# tests/test_rows_chunked.py
"""linear_rows_chunked == linear_rows: the node-chunked edge Jacobian (one chunk's J at a time,
so a >2.8k-atom cell never builds a >2^31-element tensor) gives the same design rows."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")


def _problem(n_cfg=8, per_batch=4):
    from ace_jax.eval import load
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    cfgs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")
    big = [c for c in cfgs if len(c.numbers) == 64][:1]           # a 64-atom cell among small ones
    cfgs = big + [c for c in cfgs if len(c.numbers) < 64][: n_cfg - 1]
    ds = build_dataset(cfgs, meta, np.asarray(z["E0"]), per_batch)
    g = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                 NZ=len(meta["elements"]), C=per_batch)
    return model, g, ds


@pytest.mark.parametrize("chunk", [1, 7, 64, 10_000])
def test_chunked_rows_equal_unchunked(chunk):
    from ace_jax.fit.rows import linear_rows, linear_rows_chunked
    model, g, ds = _problem()
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        ref = linear_rows(model, g, b)[0]
        got = linear_rows_chunked(model, g, b, node_chunk=chunk)
        for k in ("E", "F", "V"):
            np.testing.assert_allclose(np.asarray(getattr(got, k)), np.asarray(getattr(ref, k)),
                                       rtol=1e-12, atol=1e-12, err_msg=f"{k} chunk={chunk}")


@pytest.mark.slow
def test_chunked_rows_big_cell():
    """A 4096-atom diamond Si cell: chunked rows are finite and satisfy the force/energy
    consistency sum_i F_i = 0 row-wise (translation invariance), without building the
    whole-cell edge Jacobian."""
    from ase.build import bulk
    from ace_jax.eval import load
    from ace_jax.fit.data import Config, build_dataset
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.rows import linear_rows_chunked
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((8, 8, 8))
    at.rattle(0.05, seed=1)
    c = Config(at.positions, at.numbers, at.cell.array, at.pbc, None, None, None, 1.0, 1.0, 1.0)
    ds = build_dataset([c], meta, np.asarray(z["E0"]), 1)
    g = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"], NZ=1, C=1)
    r = linear_rows_chunked(model, g, jax.tree.map(lambda a: a[0], ds), node_chunk=512)
    F = np.asarray(r.F)
    assert np.all(np.isfinite(F))
    np.testing.assert_allclose(F.sum(axis=0), 0.0, atol=1e-8)
