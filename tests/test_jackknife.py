import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg
from ase.build import bulk
from conftest import _orders


def _cfg(at):
    from ace_jax.fit.data import Config
    return Config(at.get_positions(), at.get_atomic_numbers(), at.get_cell().array, at.get_pbc(),
                  0.0, np.zeros((len(at), 3)), np.zeros((3, 3)), 1.0, 1.0, 1.0)


def test_config_blocks_small_and_large():
    from ace_jax.fit.clusters import config_blocks
    ell = 3 * 6.25
    assert config_blocks(_cfg(bulk("Ni", "fcc", a=3.65, cubic=True).repeat(4)), ell) is None   # 14.6 A
    big = bulk("Ni", "fcc", a=3.65, cubic=True).repeat((12, 4, 4))                              # 43.8 A in x
    b = config_blocks(_cfg(big), ell)
    frac = big.get_scaled_positions(wrap=True)[:, 0]
    assert len(np.unique(b)) == 2                                                               # floor(43.8/18.75)
    assert len({(int(fx * 2), int(bi)) for fx, bi in zip(frac, b)}) == 2                        # id = f(floor(2 fx))


def test_config_blocks_nonperiodic_uses_bounding_box():
    from ace_jax.fit.clusters import config_blocks
    at = bulk("Ni", "fcc", a=3.65, cubic=True).repeat((16, 16, 2))                              # 58.4 A in x, y
    at.pbc = (False, False, True)
    at.center(vacuum=10.0, axis=(0, 1))
    assert len(np.unique(config_blocks(_cfg(at), 3 * 6.25))) == 9                               # 3 x 3 x 1


def test_row_clusters_ids():
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.data import build_dataset
    small = _cfg(bulk("Ni", "fcc", a=3.65, cubic=True).repeat(2))
    large = _cfg(bulk("Ni", "fcc", a=3.65, cubic=True).repeat((12, 4, 4)))
    ds = build_dataset([small, large], {"elements": [28], "rcut": 6.25}, np.zeros(1), 2)
    rc, K = row_clusters(ds, [small, large], 3 * 6.25)
    b, ns, nl = rc[0], len(small.numbers), len(large.numbers)
    assert b["E"][0] == b["V"][0] == 0 and np.all(b["F"][:ns] == 0)                            # small: one id
    fl = b["F"][ns:ns + nl]
    assert set(fl.tolist()) == {1, 2} and b["E"][1] == b["V"][1] == 3 and K == 4               # blocks, then E/V
    assert np.all(b["F"][ns + nl:] == -1)
    rc_inf, K_inf = row_clusters(ds, None, float("inf"))
    assert K_inf == 2 and np.all(rc_inf[0]["F"][ns:ns + nl] == 1)


def test_config_blocks_degenerate_cell_uses_cartesian_bounding_box():
    from ace_jax.fit.clusters import config_blocks
    for pbc in (False, True):
        at = bulk("Ni", "fcc", a=3.65, cubic=True).repeat((12, 4, 4))
        at.set_cell(np.zeros((3, 3)))
        at.pbc = pbc
        b = config_blocks(_cfg(at), 3 * 6.25)
        assert b is not None and len(np.unique(b)) == 2
        x = at.get_positions()[:, 0]
        assert len({(int(xi > x.min() + (x.max() - x.min()) / 2), int(bi)) for xi, bi in zip(x, b)}) == 2


def test_row_clusters_validates_configs():
    import pytest
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.data import build_dataset
    small = _cfg(bulk("Ni", "fcc", a=3.65, cubic=True).repeat(2))
    ds = build_dataset([small], {"elements": [28], "rcut": 6.25}, np.zeros(1), 1)
    with pytest.raises(ValueError):
        row_clusters(ds, None, 18.75)
    with pytest.raises(ValueError):
        row_clusters(ds, [small, small], 18.75)


def _rows(prob, ds, sig, rc=None):
    """Dense whitened rows Psi (n, L) and targets y~ (n,), with each row's label: the global live-config
    index, or (rc given) the row's cluster id from row_clusters."""
    from ace_jax.fit.rows import linear_rows
    P, Y, cf = [], [], []
    g = 0
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a, i=i: a[i], ds)
        r = linear_rows(prob.model, prob.cfg, b)[0]
        for c in np.flatnonzero(np.asarray(b.cfg_mask)):
            lab = (lambda kind, j, i=i: rc[i][kind][j]) if rc is not None else (lambda kind, j, g=g: g)
            w = float(b.w_E[c]) / sig[0]
            P.append(np.asarray(r.E[c]) * w); Y.append(float(b.y_E[c]) * w); cf.append(lab("E", c))
            for n in np.flatnonzero(np.asarray(b.node_cfg) == c):
                w = float(b.w_F[n]) / sig[1]
                for a in range(3):
                    P.append(np.asarray(r.F[n, a]) * w); Y.append(float(b.y_F[n, a]) * w); cf.append(lab("F", n))
            w = float(b.w_V[c]) / sig[2]
            for v in range(6):
                P.append(np.asarray(r.V[c, v]) * w); Y.append(float(b.y_V[c, v]) * w); cf.append(lab("V", c))
            g += 1
    return np.array(P), np.array(Y), np.array(cf)


# Reference-solve noise floor, as a fraction of max|c|.  The tiny fixture's A has cond ~ 7e11 (S: 1e10;
# 75 rows for L = 120 columns, so the prior sets the small end), and a float64 dense refit c - c_(-k)
# carries ~1e-8 max|c| of solve noise (checked against a 40-digit mpmath refit: dense f64 refit 1.1e-8,
# press_scores 2.7e-9).  An algebra error would show at |c - c_(-k)| / max|c| ~ 1e-2.
_REFIT_ATOL = 1e-7


def _A(post):
    D = 1.0 / np.asarray(post.dinv)
    S = np.asarray(post.chol) @ np.asarray(post.chol).T
    return S * D[:, None] * D[None, :]


def test_press_equals_exact_deletion(ard_setup):
    """c - c_(-k) = A^-1 g~_k for every whole-configuration cluster, h fixed."""
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores
    prob, ds, ev, h, post = ard_setup
    sig = ev.sigmas(h)
    Psi, y, cf = _rows(prob, ds, sig)
    A = _A(post)
    c = np.linalg.solve(A, Psi.T @ y)
    np.testing.assert_allclose(c, post.mean, rtol=1e-7, atol=_REFIT_ATOL * np.abs(c).max())
    rc, K = row_clusters(ds, None, float("inf"))
    G, _ = press_scores(post, prob, ds, rc, K, sig)
    assert K == cf.max() + 1
    for k in range(K):
        m = cf == k
        ck = np.linalg.solve(A - Psi[m].T @ Psi[m], Psi[~m].T @ y[~m])
        np.testing.assert_allclose(c - ck, np.linalg.solve(A, G[:, k]), rtol=1e-6, atol=_REFIT_ATOL * np.abs(c).max())


def test_push_through_matches_exact(ard_setup):
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores
    prob, ds, ev, h, post = ard_setup
    rc, K = row_clusters(ds, None, float("inf"))
    Ge, _ = press_scores(post, prob, ds, rc, K, ev.sigmas(h), mode="exact")
    Gp, _ = press_scores(post, prob, ds, rc, K, ev.sigmas(h), mode="pushthrough")
    np.testing.assert_allclose(Gp, Ge, rtol=1e-7, atol=1e-10 * np.abs(Ge).max())


def test_press_high_leverage_cluster(ard_setup):
    """Prior precision > 0 keeps every lambda_max(H_kk) < 1; a near-1 cluster stays finite and exact."""
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores
    prob, ds, ev, h, post = ard_setup
    rc, K = row_clusters(ds, None, float("inf"))
    G, lev = press_scores(post, prob, ds, rc, K, ev.sigmas(h))
    assert np.all(lev >= 0) and np.all(lev < 1) and np.isfinite(G).all()
    hh = np.array(h, float)
    hh[1] -= 3.0                                                                # sigma_F / e^3: data dominate
    post2 = post._replace(**_posterior_at(ev, hh, post))
    G2, lev2 = press_scores(post2, prob, ds, rc, K, ev.sigmas(hh))
    assert lev2.max() > lev.max() and np.all(lev2 < 1) and np.isfinite(G2).all()
    Psi, y, cf = _rows(prob, ds, ev.sigmas(hh))                                  # exact at high leverage too
    A = _A(post2)
    c = np.linalg.solve(A, Psi.T @ y)
    for k in range(K):
        m = cf == k
        ck = np.linalg.solve(A - Psi[m].T @ Psi[m], Psi[~m].T @ y[~m])
        np.testing.assert_allclose(c - ck, np.linalg.solve(A, G2[:, k]), rtol=1e-6, atol=_REFIT_ATOL * np.abs(c).max())


def _posterior_at(ev, h, post):
    from ace_jax.fit.ard import ard_posterior
    p = ard_posterior(ev, h, 2.0, post.meta)
    return {"mean": p.mean, "chol": p.chol, "dinv": p.dinv}


def test_press_spatial_and_ev_clusters(tiny_linear_problem):
    from ace_jax.fit.ard import ARDEvidence, ard_posterior, ard_statistics, body_order_columns
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.data import Config, build_dataset
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.jackknife import press_scores
    prob, _ = tiny_linear_problem
    rcut = float(prob.cfg.rcut)
    ell = 1.2 * rcut
    n = int(np.ceil(2 * ell / 5.43))                   # width in [2 ell, 2 ell + a): exactly 2 blocks
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((n, 1, 1))
    at.rattle(0.05, seed=0)
    rng = np.random.default_rng(0)
    cfg = Config(at.get_positions(), at.get_atomic_numbers(), at.get_cell().array, at.get_pbc(),
                 float(rng.normal()), rng.normal(size=(len(at), 3)), rng.normal(size=(3, 3)), 1e-3, 1.0, 1e-3)
    # w_E = w_V = 1e-3: at unit weights this lone config's energy row is the only energy information, so
    # the E/V cluster has 1 - lambda_max(H_kk) = 1.4e-10 and deleting it is ill-posed at the 1e-5 level in
    # float64 (both the dense refit and PRESS, vs a 40-digit refit).  At 1e-3 it is still the high-leverage
    # cluster (lambda_max = 0.9999) but both sides agree with the 40-digit refit to 1e-11 max|c|.
    ds = build_dataset([cfg], {"elements": [14], "rcut": rcut}, np.zeros(1), 1)
    meta = {"nnll": [[None] * o for o in _orders(prob)], "n_B": prob.cfg.n_B, "n_pair": prob.cfg.n_pair,
            "NZ": prob.cfg.NZ, "rcut": rcut, "elements": [14]}
    theta = default_prior(2.35).mu
    ev = ARDEvidence(ard_statistics(theta, prob, ds, "joint"), np.asarray(prob.gamma),
                     body_order_columns(meta, prob.cfg))
    h = ev.h0(theta)
    post = ard_posterior(ev, h, 2.0, meta)
    rc, K = row_clusters(ds, [cfg], ell)
    assert K == 3
    Psi, y, kid = _rows(prob, ds, ev.sigmas(h), rc=rc)
    A = _A(post)
    c = np.linalg.solve(A, Psi.T @ y)
    G, _ = press_scores(post, prob, ds, rc, K, ev.sigmas(h))
    for k in range(K):
        m = kid == k
        ck = np.linalg.solve(A - Psi[m].T @ Psi[m], Psi[~m].T @ y[~m])
        np.testing.assert_allclose(c - ck, np.linalg.solve(A, G[:, k]), rtol=1e-6, atol=1e-9 * np.abs(c).max())


def test_centring_and_factor(ard_setup):
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import atom_shape, press_scores, shape_factor
    from ace_jax.fit.rows import linear_rows
    prob, ds, ev, h, post = ard_setup
    rc, K = row_clusters(ds, None, float("inf"))
    G, _ = press_scores(post, prob, ds, rc, K, ev.sigmas(h))
    R = np.asarray(shape_factor(post, G))
    # S is defined by its factor: solve with it (LAPACK potrs).  Re-factorising S = L L^T (cond 1e10), by
    # LU or Cholesky, alone moves Qt by 4e-7 -- far above the 1e-8 target.
    Qt = scipy.linalg.cho_solve((np.asarray(post.chol), True),
                                (G - G.mean(1, keepdims=True)) * np.asarray(post.dinv)[:, None])
    M = Qt @ Qt.T
    np.testing.assert_allclose(R @ R.T, M, rtol=1e-8, atol=1e-12 * np.abs(M).max())
    Rt = np.asarray(shape_factor(post, G, tau=0.9))
    assert Rt.shape[1] <= min(K, len(post.mean)) and np.trace(Rt @ Rt.T) >= 0.9 * np.trace(M) * (1 - 1e-12)
    Fr = np.asarray(linear_rows(prob.model, prob.cfg, jax.tree.map(lambda a: a[0], ds))[0].F)
    V = atom_shape(jnp.asarray(R), post.dinv, Fr)
    u = Fr * np.asarray(post.dinv)[None, None, :]
    Vref = np.einsum("nal,lm,nbm->nab", u, M, u)
    np.testing.assert_allclose(V, Vref, rtol=1e-8, atol=1e-14 * max(np.abs(Vref).max(), 1e-300))
    np.testing.assert_allclose(V, np.swapaxes(V, 1, 2))


def _row_block_clusters(ds):
    """Every row-block (an E row, an atom's F triple, a config's V sextet) its own cluster, ids contiguous
    within each batch and increasing across batches."""
    out, k = [], 0
    for i in range(ds.n_batches):
        cm, nc = np.asarray(ds.cfg_mask[i]), np.asarray(ds.node_cfg[i])
        E, F, V = np.full(len(cm), -1), np.full(len(nc), -1), np.full(len(cm), -1)
        for c in np.flatnonzero(cm):
            E[c] = k; k += 1
            for n in np.flatnonzero(nc == c):
                F[n] = k; k += 1
            V[c] = k; k += 1
        out.append({"E": E, "F": F, "V": V})
    return out, k


def test_press_block_mode(ard_setup):
    """Block mode is exact when every cluster is one row-block, and an approximation (finite, same order,
    not identical) for whole-configuration clusters."""
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores
    prob, ds, ev, h, post = ard_setup
    sig = ev.sigmas(h)
    rb, Kb = _row_block_clusters(ds)
    Ge, le = press_scores(post, prob, ds, rb, Kb, sig, mode="exact")
    Gb, lb = press_scores(post, prob, ds, rb, Kb, sig, mode="block")
    np.testing.assert_allclose(Gb, Ge, rtol=1e-8, atol=1e-10 * np.abs(Ge).max())
    np.testing.assert_allclose(lb, le)
    rc, K = row_clusters(ds, None, float("inf"))
    Ge, _ = press_scores(post, prob, ds, rc, K, sig, mode="exact")
    Gb, _ = press_scores(post, prob, ds, rc, K, sig, mode="block")
    assert np.isfinite(Gb).all() and not np.allclose(Gb, Ge)
    rel = np.linalg.norm(Gb - Ge) / np.linalg.norm(Ge)
    assert rel < 1.0


def test_press_rejects_cluster_spanning_batches(ard_setup):
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores
    prob, ds, ev, h, post = ard_setup
    rc, K = row_clusters(ds, None, float("inf"))
    assert ds.n_batches >= 2
    bad = [dict(b) for b in rc]
    first = int(np.max(bad[0]["E"]))
    bad[1] = {q: np.where(v >= 0, first, v) for q, v in bad[1].items()}             # batch 1 reuses an id
    with pytest.raises(ValueError, match="spans batches"):
        press_scores(post, prob, ds, bad, K, ev.sigmas(h))


def test_shape_factor_zero_spread_gives_zero_column(ard_setup):
    """K = 1 (or all-zero scores) centre to zero: R is a single zero column for any tau."""
    from ace_jax.fit.jackknife import shape_factor
    _, _, _, _, post = ard_setup
    L = len(post.mean)
    G = np.random.default_rng(0).normal(size=(L, 1))
    for tau in (1.0, 0.9):
        R = np.asarray(shape_factor(post, G, tau=tau))
        assert R.shape == (L, 1) and np.all(R == 0)
    R = np.asarray(shape_factor(post, np.zeros((L, 3)), tau=0.9))
    assert R.shape == (L, 1) and np.all(R == 0)
