"""Linear rows against ACEfit's design matrix for the same model (row-for-row);
residual force/virial rows against autodiff of the residual energy rows."""
import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import VOIGT, build_dataset, flat_edges, load_configs
from ace_jax.fit.hypers import Hypers
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.rows import batch_rows, linear_rows, residual_rows

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
DESIGN = FIXTURE_DIR / "si_tiny_design.npz"
pytestmark = pytest.mark.skipif(not (XYZ.exists() and DESIGN.exists()), reason="missing GP fixtures")

THETA = Hypers(log_ell=np.log(0.8), log_A=np.log(0.3), log_alpha=np.log(1.2), log_r0=np.log(2.35),
               log_eps=np.log(0.4), log_rho=np.log(4.0), log_sigma_c=0.0,
               log_sigma_E=0.0, log_sigma_F=0.0, log_sigma_V=0.0)


@pytest.fixture(scope="module")
def setup():
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")
    C = 4
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=C)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=C)
    X, S = site_features(model, cfg, ds)
    scale = descriptor_scale(X, ds.node_mask)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 6, scale)
    return model, meta, configs, ds, cfg, ind


def test_linear_rows_match_acefit_design_matrix(setup):
    model, meta, configs, ds, cfg, ind = setup
    d = np.load(DESIGN)
    A, natoms = d["A"], d["natoms"]
    # ACEfit emits virial rows only for configs that carry a virial (configs[0],
    # the isolated atom, has none): rows per config = 1 + 3 n + 6 has_virial.
    has_virial = np.array([c.virial is not None for c in configs])
    offs = np.concatenate([[0], np.cumsum(1 + 3 * natoms + 6 * has_virial)])
    assert offs[-1] == A.shape[0]
    L = cfg.len_basis
    with highest_precision():
        for b in range(ds.n_batches):
            batch = jax.tree.map(lambda a: a[b], ds)
            rows, X, J = linear_rows(model, cfg, batch)
            for c in range(cfg.C):
                g = b * cfg.C + c
                if g >= len(configs):
                    break
                n = natoms[g]
                ref = A[offs[g]:offs[g + 1]]
                nodes = np.flatnonzero(np.asarray(batch.node_cfg) == c)
                assert len(nodes) == n
                E = np.asarray(rows.E[c, :L])
                F = np.asarray(rows.F[nodes, :, :L]).reshape(3 * n, L)
                V = np.asarray(rows.V[c, :, :L])
                got = np.concatenate([E[None], F] + ([V] if has_virial[g] else []))
                assert np.abs(got - ref).max() < 1e-8 * max(1.0, np.abs(ref).max())


# Both kernel families, with and without the SE magnitude factor.  The reference rows
# must be non-trivial for the comparison to mean anything: before Rulings
# R18/R19 (RMS distance; rho as decay length) the bump was ~1e-13 at the
# fixture's nearest-inducing distances and the check compared zeros to zeros.
@pytest.mark.parametrize("kind, bump", [("matern32", True), ("cosine", False)])
def test_residual_rows_match_autodiff_of_energy_rows(setup, kind, bump):
    model, meta, configs, ds, cfg, ind = setup
    spec = KernelSpec(kind=kind, bump=bump, D=cfg.D)
    batch = jax.tree.map(lambda a: a[0], ds)
    L, M = cfg.len_basis, ind.XM.shape[0]

    Ncap, K = batch.nbr.shape

    with highest_precision():
        rows, X, J = linear_rows(model, cfg, batch)
        res = residual_rows(THETA, spec, ind, cfg, batch, X, J)

    def e_res(rij):
        # energy rows only need X and s; skip the (expensive) Jacobian inside linear_rows
        b = batch._replace(rij=rij)
        r, send, recv, m = flat_edges(rij, batch.nbr, batch.nbr_mask)
        Xr = model.compact_basis(r, batch.node_z[send], batch.node_z[recv], send, Ncap, m)
        return residual_rows(THETA, spec, ind, cfg, b, Xr, J).E          # (C, M)

    with highest_precision():
        T = jax.jacrev(e_res)(batch.rij)                                  # (C, M, Ncap, K, 3)
    node_cfg = np.asarray(batch.node_cfg); nbr = np.asarray(batch.nbr).reshape(-1)
    own = np.minimum(node_cfg, cfg.C - 1)                                 # padded nodes -> any slot, masked below
    Tn = np.asarray(T)[own, :, np.arange(Ncap)]                           # (Ncap, M, K, 3): own-config slice
    Tn = np.where(np.asarray(batch.node_mask)[:, None, None, None], Tn, 0.0)
    dEdr = np.zeros((Ncap, 3, M))
    dEdr -= Tn.sum(2).transpose(0, 2, 1)                                  # d/dr_i (sender)
    np.add.at(dEdr, nbr, Tn.transpose(0, 2, 3, 1).reshape(Ncap * K, 3, M))  # d/dr_j (receiver)
    assert np.abs(dEdr).max() > 1e-6, "reference rows vanished: the check is vacuous"
    assert np.abs(np.asarray(res.F) + dEdr).max() < 1e-9 * max(1.0, np.abs(dEdr).max())
    rij = np.asarray(batch.rij).reshape(Ncap * K, 3)
    Te = Tn.transpose(0, 2, 3, 1).reshape(Ncap * K, 3, M)
    V = np.zeros((cfg.C + 1, 6, M))
    ecfg = node_cfg[np.repeat(np.arange(Ncap), K)]
    for v, (a, b_) in enumerate(VOIGT):
        np.add.at(V[:, v], ecfg, -0.5 * (Te[:, a] * rij[:, b_, None] + Te[:, b_] * rij[:, a, None]))
    assert np.abs(np.asarray(res.V) - V[:cfg.C]).max() < 1e-9 * max(1.0, np.abs(V).max())


def test_residual_rows_density_map_match_autodiff(setup):
    """Density feature map (sqrt of the ACE pair-density channels, the FS/PACE
    embedding): the residual force/virial rows are still the exact derivative
    of the residual energy rows -- the JU contraction folds Pmap and the sqrt
    warp into the edge Jacobian correctly."""
    from ace_jax.fit.inducing import build_pmap, select_inducing
    model, meta, configs, ds, cfg, ind0 = setup
    X, S = site_features(model, cfg, ds)
    scale = descriptor_scale(X, ds.node_mask)
    Pmap = build_pmap(cfg, scale, density="pair")
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 6, scale, Pmap=Pmap, warp="sqrt")
    assert ind.XM.shape[1] == cfg.n_pair and ind.warp == "sqrt"
    spec = KernelSpec(kind="matern32", bump=True, D=cfg.D)
    batch = jax.tree.map(lambda a: a[0], ds)
    Ncap, K = batch.nbr.shape
    M = ind.XM.shape[0]
    with highest_precision():
        rows, X0, J = linear_rows(model, cfg, batch)
        res = residual_rows(THETA, spec, ind, cfg, batch, X0, J)
    assert bool(jnp.all(jnp.isfinite(res.F))) and bool(jnp.all(jnp.isfinite(res.V)))

    def e_res(rij):
        b = batch._replace(rij=rij)
        r, send, recv, m = flat_edges(rij, batch.nbr, batch.nbr_mask)
        Xr = model.compact_basis(r, batch.node_z[send], batch.node_z[recv], send, Ncap, m)
        return residual_rows(THETA, spec, ind, cfg, b, Xr, J).E

    with highest_precision():
        T = jax.jacrev(e_res)(batch.rij)
    node_cfg = np.asarray(batch.node_cfg); nbr = np.asarray(batch.nbr).reshape(-1)
    own = np.minimum(node_cfg, cfg.C - 1)
    Tn = np.asarray(T)[own, :, np.arange(Ncap)]
    Tn = np.where(np.asarray(batch.node_mask)[:, None, None, None], Tn, 0.0)
    dEdr = np.zeros((Ncap, 3, M))
    dEdr -= Tn.sum(2).transpose(0, 2, 1)
    np.add.at(dEdr, nbr, Tn.transpose(0, 2, 3, 1).reshape(Ncap * K, 3, M))
    assert np.abs(dEdr).max() > 1e-6, "reference rows vanished: the check is vacuous"
    assert np.abs(np.asarray(res.F) + dEdr).max() < 1e-9 * max(1.0, np.abs(dEdr).max())


def test_batch_rows_concatenates(setup):
    model, meta, configs, ds, cfg, ind = setup
    spec = KernelSpec(kind="cosine", bump=False, D=cfg.D)
    batch = jax.tree.map(lambda a: a[0], ds)
    with highest_precision():
        rows = jax.jit(lambda th: batch_rows(th, spec, model, ind, cfg, batch))(THETA)
    assert rows.E.shape == (cfg.C, cfg.len_basis + ind.XM.shape[0])
    assert bool(jnp.all(jnp.isfinite(rows.F))) and bool(jnp.all(jnp.isfinite(rows.V)))


def test_edge_jacobian_dense_matches_sparse(setup):
    """The dense per-node contraction equals the edge-gather form on a batch."""
    model, meta, configs, ds, cfg, ind = setup
    batch = jax.tree.map(lambda a: a[0], ds)
    Ncap, K = batch.nbr.shape
    rij, send, recv, mask = flat_edges(batch.rij, batch.nbr, batch.nbr_mask)
    with highest_precision():
        Xs, Js = model.edge_jacobian(rij, batch.node_z[send], batch.node_z[recv], send, Ncap, mask)
        Xd, Jd = model.edge_jacobian_dense(batch.rij, jnp.broadcast_to(batch.node_z[:, None], (Ncap, K)),
                                           batch.node_z[batch.nbr], batch.nbr_mask)
    assert float(jnp.abs(Xs - Xd).max()) < 1e-11 * max(1.0, float(jnp.abs(Xs).max()))
    assert float(jnp.abs(Js - Jd).max()) < 1e-11 * max(1.0, float(jnp.abs(Js).max()))
