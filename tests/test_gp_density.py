"""Per-species FS-density columns (ace_jax.fit.density): force/virial rows are the
exact derivative of the energy rows (analytic, from the edge Jacobian)."""
import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from conftest import FIXTURE_DIR
from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import VOIGT, build_dataset, flat_edges, load_configs
from ace_jax.fit.inducing import GPConfig
from ace_jax.fit.rows import linear_rows
from ace_jax.fit.density import density_rows

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing fixtures")


def test_density_rows_match_autodiff_of_energy():
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), 4)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=4)
    batch = jax.tree.map(lambda a: a[0], ds)
    Ncap, K = batch.nbr.shape
    NZ = cfg.NZ
    eta = jnp.asarray(np.random.default_rng(0).standard_normal((NZ, cfg.n_pair)))
    with highest_precision():
        rows, X, J = linear_rows(model, cfg, batch)
        dr = density_rows(eta, cfg, batch, X, J)

        def e_dens(rij):
            b = batch._replace(rij=rij)
            r, send, recv, m = flat_edges(rij, batch.nbr, batch.nbr_mask)
            Xr = model.compact_basis(r, batch.node_z[send], batch.node_z[recv], send, Ncap, m)
            return density_rows(eta, cfg, b, Xr, J).E
        T = jax.jacrev(e_dens)(batch.rij)
    node_cfg = np.asarray(batch.node_cfg); nbr = np.asarray(batch.nbr).reshape(-1)
    own = np.minimum(node_cfg, cfg.C - 1)
    Tn = np.where(np.asarray(batch.node_mask)[:, None, None, None],
                  np.asarray(T)[own, :, np.arange(Ncap)], 0.0)               # (Ncap, NZ, K, 3)
    dEdr = np.zeros((Ncap, 3, NZ))
    dEdr -= Tn.sum(2).transpose(0, 2, 1)
    np.add.at(dEdr, nbr, Tn.transpose(0, 2, 3, 1).reshape(Ncap * K, 3, NZ))
    assert np.abs(dEdr).max() > 1e-3
    assert np.abs(np.asarray(dr.F) + dEdr).max() < 1e-9 * max(1.0, np.abs(dEdr).max())

    rij = np.asarray(batch.rij).reshape(Ncap * K, 3); Te = Tn.transpose(0, 2, 3, 1).reshape(Ncap * K, 3, NZ)
    Vref = np.zeros((cfg.C + 1, 6, NZ)); ecfg = node_cfg[np.repeat(np.arange(Ncap), K)]
    for v, (a, b_) in enumerate(VOIGT):
        np.add.at(Vref[:, v], ecfg, -0.5 * (Te[:, a] * rij[:, b_, None] + Te[:, b_] * rij[:, a, None]))
    assert np.abs(np.asarray(dr.V) - Vref[:cfg.C]).max() < 1e-9 * max(1.0, np.abs(Vref).max())
