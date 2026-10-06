"""--uq ard-gp: the ARD jackknife-sandwich posterior over the GP arm's joint design [B | k(B, B_M)] at the
fixed theta_MAP (docs/dev/specs/2026-10-05-gp-discrepancy-design.md, Phase 3)."""
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ace_jax.eval import highest_precision  # noqa: E402


def _meta(prob):
    from conftest import _orders
    return {"nnll": [[None] * o for o in _orders(prob)], "n_B": prob.cfg.n_B, "n_pair": prob.cfg.n_pair,
            "NZ": prob.cfg.NZ, "rcut": prob.cfg.rcut, "elements": [14]}


@pytest.fixture
def gp_ev(tiny_gp_problem):
    from ace_jax.fit.ard import (ARDEvidence, ard_gamma, ard_statistics, body_order_columns,
                                 gp_body_columns)
    from ace_jax.fit.prior_root import prior_root
    prob, ds, theta = tiny_gp_problem
    M = prob.ind.XM.shape[0]
    with highest_precision():
        st = ard_statistics(theta, prob, ds, "joint", columns="joint")
        ev = ARDEvidence(st, ard_gamma(prob), gp_body_columns(body_order_columns(_meta(prob), prob.cfg), M),
                         root=prior_root(prob, theta))
    return prob, ds, theta, ev


def test_joint_statistics_have_the_inducing_columns(gp_ev):
    prob, _, _, ev = gp_ev
    L, M = prob.cfg.len_basis, prob.ind.XM.shape[0]
    assert ev._data[0][1].shape == (L + M, L + M)          # G_F over the joint design
    assert ev.groups[0] == -1 and len(ev.groups) == 1 + len(set(ev.body_col[:L]) - {0})


def _gp_dense_rows(prob, ds, theta, sig):
    """Whitened joint rows Psi (n, L+M), targets y~ and the live-configuration index of each row, in
    batch / configuration / (E, F x 3 per atom, V x 6) order.  Rows of zero structural weight (a
    configuration without virials, say) are absent, as in the statistics."""
    from ace_jax.fit.rows import batch_rows
    P, Y, cf, g = [], [], [], 0

    def add(phi, yv, w):
        if w > 0:
            P.append(np.asarray(phi) * w); Y.append(float(yv) * w); cf.append(g)

    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a, i=i: a[i], ds)
        r = batch_rows(theta, prob.spec, prob.model, prob.ind, prob.cfg, b)
        for c in np.flatnonzero(np.asarray(b.cfg_mask)):
            add(r.E[c], b.y_E[c], float(b.w_E[c]) / sig[0])
            for n in np.flatnonzero(np.asarray(b.node_cfg) == c):
                for a in range(3):
                    add(r.F[n, a], b.y_F[n, a], float(b.w_F[n]) / sig[1])
            for v in range(6):
                add(r.V[c, v], b.y_V[c, v], float(b.w_V[c]) / sig[2])
            g += 1
    return np.array(P), np.array(Y), np.array(cf)


def _obs_space_reference(prob, ds, theta):
    """(LML, posterior mean) of the hybrid model at theta from the observation-space covariance
    C~ = Psi Lambda^-1 Psi^T + I (Lambda = blkdiag(Gamma^2/sigma_c^2, K_MM)): independent of the
    weight-space code, and well-conditioned (the identity), unlike G + Lambda on this fixture."""
    from ace_jax.fit.kernels import K_MM
    from ace_jax.fit.stats import sufficient_statistics
    sig = np.exp([float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"])
    Psi, y, _ = _gp_dense_rows(prob, ds, theta, sig)
    ind = prob.ind
    L = prob.cfg.len_basis
    Kinv = np.linalg.inv(np.asarray(K_MM(theta, prob.spec, ind.XM, ind.SM, ind.ZM, ind.embed)))
    lin_var = float(np.exp(2 * theta.log_sigma_c)) / np.asarray(prob.gamma) ** 2
    PL = Psi[:, :L] * lin_var[None, :]
    PG = Psi[:, L:] @ Kinv
    C = PL @ Psi[:, :L].T + PG @ Psi[:, L:].T + np.eye(len(y))
    a = np.linalg.solve(C, y)
    st = sufficient_statistics(theta, prob.spec, prob.model, ind, prob.cfg, ds)
    logtau = sum(float(getattr(st, f"logw_{q}")) - float(getattr(st, f"n_{q}")) * 2 * np.log(sig[i])
                 for i, q in enumerate("EFV"))
    lml = -0.5 * y @ a - 0.5 * np.linalg.slogdet(C)[1] - 0.5 * len(y) * np.log(2 * np.pi) + 0.5 * logtau
    return lml, np.concatenate([PL.T @ a, PG.T @ a]), st


def test_evidence_at_h0_is_the_gp_lml(gp_ev):
    """h0 (a_k = -2 log sigma_c on the body orders, a_GP = 0, the MAP's sigma_q) reproduces the hybrid
    model's log marginal likelihood at theta: the ARD prior at h0 IS blkdiag(Gamma^2/sigma_c^2, K_MM).
    ARDEvidence's constant leaves out the structural log weights and the 2 pi term:
    LML = logev + 1/2 sum_q logw_q - 1/2 N log(2 pi)."""
    prob, ds, theta, ev = gp_ev
    with highest_precision():
        lml, _, st = _obs_space_reference(prob, ds, theta)
        v, _ = ev.value_and_grad(ev.h0(theta))
    N = sum(float(getattr(st, f"n_{q}")) for q in "EFV")
    logw = sum(float(getattr(st, f"logw_{q}")) for q in "EFV")
    assert v + 0.5 * logw - 0.5 * N * np.log(2 * np.pi) == pytest.approx(lml, rel=1e-8)


def test_posterior_mean_at_h0_is_the_gp_map_mean(gp_ev):
    """The linear coefficients to 1e-6; the inducing weights through what they predict (Psi mean): on their
    own they are only as determined as K_MM is conditioned (cond 7e8 here), in both the reference and S."""
    from ace_jax.fit.ard import ard_posterior
    prob, ds, theta, ev = gp_ev
    L = prob.cfg.len_basis
    with highest_precision():
        _, mu, _ = _obs_space_reference(prob, ds, theta)
        post = ard_posterior(ev, ev.h0(theta), 1.0, _meta(prob), theta=theta)
        Psi, _, _ = _gp_dense_rows(prob, ds, theta, np.exp([float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"]))
    np.testing.assert_allclose(post.mean[:L], mu[:L], rtol=1e-6, atol=1e-8 * np.abs(mu[:L]).max())
    np.testing.assert_allclose(Psi @ post.mean, Psi @ mu, rtol=1e-6, atol=1e-8 * np.abs(Psi @ mu).max())


def test_gp_evidence_gradient_matches_finite_differences(gp_ev):
    _, _, theta, ev = gp_ev
    h = ev.h0(theta) + 0.1
    _, g = ev.value_and_grad(h)
    for i in range(len(h)):
        e = np.zeros_like(h)
        e[i] = 1e-3          # 1e-5 is roundoff-limited on the noise scales (~1e-7 nats of evidence noise)
        fd = (ev.value_and_grad(h + e)[0] - ev.value_and_grad(h - e)[0]) / 2e-3
        assert g[i] == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_gp_posterior_save_load_roundtrip(gp_ev, tmp_path):
    from ace_jax.fit.ard import ARDPosterior, ard_posterior
    prob, _, theta, ev = gp_ev
    post = ard_posterior(ev, ev.h0(theta), 1.0, _meta(prob), theta=theta)
    post.save(tmp_path / "p.npz", dtype=np.float64)
    back = ARDPosterior.load(tmp_path / "p.npz")
    assert back.prior_root.M == prob.ind.XM.shape[0]
    np.testing.assert_array_equal(np.asarray(back.prior_root.U), np.asarray(post.prior_root.U))
    np.testing.assert_array_equal(back.gp_theta, post.gp_theta)
