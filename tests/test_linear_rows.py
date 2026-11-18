"""The generic linear fit (`ace_jax.fit.linear`) on rows that are not energies, forces and virials: a row
generator of two quantities with several right-hand sides, streamed to Gram and QR statistics, and the
evidence and posterior from them, against dense NumPy/SciPy references."""
import types

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.stats import multivariate_normal  # noqa: E402

from ace_jax.fit.linear import objective as lo  # noqa: E402
from ace_jax.fit.linear import stats as ls  # noqa: E402
from ace_jax.fit.linear.lml_host import HostQREvidence  # noqa: E402

L, K, NB = 7, 3, 5                                  # columns, right-hand sides, batches
NA, NBQ = 6, 9                                      # rows per batch of quantities "a" and "b"
LS = (np.log(0.3), np.log(1.7))                     # log sigma_a, log sigma_b
LIN = np.linspace(0.5, 2.0, L) ** 2                 # prior precision (diagonal)


def _data(k=K, seed=0):
    r = np.random.default_rng(seed)
    wa, wb = r.uniform(0.5, 2.0, (NB, NA)), r.uniform(0.5, 2.0, (NB, NBQ))
    wa[:, -2:] = 0.0                                # absent rows: weight 0
    wb[1] = 0.0                                     # a batch without quantity b
    tail = () if k is None else (k,)
    ds = dict(Pa=r.normal(size=(NB, NA, L)), Ya=r.normal(size=(NB, NA) + tail), wa=wa,
              Pb=r.normal(size=(NB, NBQ, L)), Yb=r.normal(size=(NB, NBQ) + tail), wb=wb)
    return jax.tree.map(jnp.asarray, ds)


def rows(b):
    """The row generator: (Phi, Y, w) per quantity."""
    return (b["Pa"], b["Ya"], b["wa"]), (b["Pb"], b["Yb"], b["wb"])


def _dense(ds):
    """Per quantity: weighted rows and targets of the live rows, stacked over batches."""
    out = []
    for P, Y, w in ((ds["Pa"], ds["Ya"], ds["wa"]), (ds["Pb"], ds["Yb"], ds["wb"])):
        P, Y, w = (np.asarray(a).reshape((-1,) + a.shape[2:]) for a in (P, Y, w))
        live = w > 0
        out.append((P[live] * w[live, None], Y[live] * (w[live] if Y.ndim == 1 else w[live, None]), w[live]))
    return out


def _reference_evidence(ds, col=None):
    """sum over right-hand sides of log N(w y / sigma; 0, I + A Lam^-1 A^T) + sum log(w / sigma), A the
    whitened rows: the evidence of y with noise precision (w / sigma)^2 and prior precision LIN."""
    parts = _dense(ds)
    A = np.concatenate([Pw / np.exp(s) for (Pw, _, _), s in zip(parts, LS)])
    T = np.concatenate([(yw if yw.ndim == 2 else yw[:, None]) / np.exp(s) for (_, yw, _), s in zip(parts, LS)])
    jac = sum(np.sum(np.log(w / np.exp(s))) for (_, _, w), s in zip(parts, LS))
    cov = np.eye(A.shape[0]) + (A / LIN) @ A.T
    cols = range(T.shape[1]) if col is None else [col]
    return sum(multivariate_normal(np.zeros(A.shape[0]), cov).logpdf(T[:, j]) + jac for j in cols)


@pytest.mark.parametrize("host", [True, False])
def test_qr_statistics_are_the_gram_of_the_rows(host):
    ds = _data()
    fac = ls.qr_statistics(rows, ds, L, labels=("a", "b"), host=host)
    gr = jax.jit(lambda d: ls.gram_statistics(rows, d))(ds)
    for f, g, (Pw, yw, w) in zip(fac, gr, _dense(ds)):
        assert f.c.shape == (L, K) and f.yy.shape == (K,)
        np.testing.assert_allclose(f.R, np.triu(f.R), atol=0)
        for got in (f.G, g.G):
            np.testing.assert_allclose(got, Pw.T @ Pw, rtol=1e-11, atol=1e-11)
        for got in (f.b, g.b):
            np.testing.assert_allclose(got, Pw.T @ yw, rtol=1e-11, atol=1e-11)
        for s in (f, g):
            np.testing.assert_allclose(s.yy, np.sum(yw * yw, 0), rtol=1e-12)
            assert float(s.n) == len(w)
            np.testing.assert_allclose(s.logw, np.sum(2 * np.log(w)), rtol=1e-12)


def test_host_and_device_streams_agree_at_every_merge_size():
    ds = _data()
    ref = ls.qr_statistics(rows, ds, L, host=True)
    for mr in (0, 4, None):
        for host in (True, False):
            got = ls.qr_statistics(rows, ds, L, merge_rows=mr, host=host)
            for a, b in zip(ref, got):                           # R is unique up to row signs: compare R^T R, R^T c
                np.testing.assert_allclose(a.G, b.G, rtol=1e-11, atol=1e-11)
                np.testing.assert_allclose(a.b, b.b, rtol=1e-11, atol=1e-11)


def test_evidence_and_posterior_match_the_dense_gaussian():
    ds = _data()
    fac = ls.qr_statistics(rows, ds, L)
    lin = jnp.asarray(LIN)
    got = float(lo.log_evidence_qr(lin, fac, LS))
    np.testing.assert_allclose(got, _reference_evidence(ds), rtol=1e-11)
    G, b, yy, logtau, N = lo.combine([f.gram() for f in fac], LS)
    chol = jnp.linalg.cholesky(G + jnp.diag(lin))
    np.testing.assert_allclose(float(lo.log_evidence(chol, b, yy, logtau, N, jnp.sum(jnp.log(lin)))), got, rtol=1e-11)
    mu, Lq = lo.posterior_qr(lin, fac, LS)
    ref = np.linalg.solve(np.asarray(G) + np.diag(LIN), np.asarray(b))
    np.testing.assert_allclose(mu, ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(lo.posterior_cholesky(chol, b)[0], ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Lq @ Lq.T, np.asarray(G) + np.diag(LIN), rtol=1e-11)


def test_one_right_hand_side_is_a_column_of_several():
    ds = _data()
    one = {**ds, "Ya": ds["Ya"][..., 1], "Yb": ds["Yb"][..., 1]}
    fk, f1 = ls.qr_statistics(rows, ds, L), ls.qr_statistics(rows, one, L)
    lin = jnp.asarray(LIN)
    for a, b in zip(fk, f1):
        assert b.c.shape == (L,) and b.yy.shape == ()
        np.testing.assert_allclose(a.b[:, 1], b.b, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(a.yy[1], b.yy, rtol=1e-12)
    np.testing.assert_allclose(lo.posterior_qr(lin, fk, LS)[0][:, 1], lo.posterior_qr(lin, f1, LS)[0], rtol=1e-10)
    np.testing.assert_allclose(float(lo.log_evidence_qr(lin, f1, LS)), _reference_evidence(ds, col=1), rtol=1e-11)
    total = sum(float(lo.log_evidence_qr(lin, [f._replace(c=f.c[:, j], yy=f.yy[j]) for f in fk], LS)) for j in range(K))
    np.testing.assert_allclose(float(lo.log_evidence_qr(lin, fk, LS)), total, rtol=1e-12)


def test_host_evidence_takes_any_quantities():
    """HostQREvidence on a plain sequence of factors, and on a statistics object that labels them: the
    value, gradient and Hessian of log_evidence_qr in (log sigma_c, log sigma_q)."""
    ds = _data(k=None)
    fac = ls.qr_statistics(rows, ds, L, labels=("a", "b"))
    gamma = np.sqrt(LIN)
    prob = types.SimpleNamespace(gamma=gamma, e0_prec=None)

    def dense(x):                                                # x = (log sigma_c, log sigma_a, log sigma_b)
        return lo.log_evidence_qr(jnp.asarray(gamma) ** 2 / jnp.exp(2.0 * x[0]), fac, [x[1], x[2]])
    x0 = jnp.asarray([0.2, *LS])
    for qs, names in ((fac, ("0", "1")), (types.SimpleNamespace(factors=fac, quantities=("a", "b")), ("a", "b"))):
        ev = HostQREvidence(qs, prob)
        assert ev.sigmas == names

        def host(x, ev=ev, names=names):
            return ev.lml(types.SimpleNamespace(log_sigma_c=x[0], **{f"log_sigma_{n}": x[i + 1] for i, n in enumerate(names)}))
        np.testing.assert_allclose(float(host(x0)), float(dense(x0)), rtol=1e-11)
        np.testing.assert_allclose(jax.grad(host)(x0), jax.grad(dense)(x0), rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(jax.hessian(host)(x0), jax.hessian(dense)(x0), rtol=1e-7, atol=1e-8)
    shared = HostQREvidence(types.SimpleNamespace(factors=fac, quantities=("a", "b")), prob, shared="b")
    x = jnp.asarray([0.2, LS[1], LS[1]])
    got = shared.lml(types.SimpleNamespace(log_sigma_c=x[0], log_sigma_b=x[1]))
    np.testing.assert_allclose(float(got), float(dense(x)), rtol=1e-11)


def test_ard_evidence_takes_any_quantities():
    """The ARD evidence (ace_jax.uq.ard) of two labelled quantities: one noise scale each, read from theta
    by label; its QR and Cholesky forms agree, and with one group at a = log of the prior precision over
    gamma^2 it is the linear evidence."""
    from ace_jax.uq.ard import ARDEvidence, ard_qr_stats, joint_ard_stats
    ds = _data(k=None)
    fac = ls.qr_statistics(rows, ds, L, labels=("a", "b"))
    labelled = types.SimpleNamespace(factors=fac, quantities=("a", "b"))
    gamma = np.sqrt(LIN)
    theta = types.SimpleNamespace(log_sigma_c=0.0, log_sigma_a=LS[0], log_sigma_b=LS[1])
    one = np.full(L, 2)
    q = ("a", "b")
    evq = ARDEvidence(ard_qr_stats(labelled), gamma, one, quantities=q)
    evc = ARDEvidence(joint_ard_stats([f.gram() for f in fac]), gamma, one, quantities=q)
    assert evq.quantities == ("a", "b") and evq._nls == 2
    h = evq.h0(theta)
    np.testing.assert_allclose(h, [*LS, 0.0])
    (vq, gq), (vc, gc) = evq.value_and_grad(h), evc.value_and_grad(h)
    np.testing.assert_allclose(vq, vc, rtol=1e-10)
    np.testing.assert_allclose(gq, gc, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(evq.sigmas(h), np.exp(LS))
    ref = float(lo.log_evidence_qr(jnp.asarray(LIN), fac, LS))            # differ by the constants ARD drops
    const = -0.5 * sum(float(f.n) for f in fac) * np.log(2 * np.pi) + 0.5 * sum(float(f.logw) for f in fac)
    np.testing.assert_allclose(vq + const, ref, rtol=1e-10)
    two = np.array([2, 2, 2, 3, 3, 3, 3])                                # two groups: one more hyperparameter
    assert len(ARDEvidence(ard_qr_stats(labelled), gamma, two, quantities=q).h0(theta)) == 4


def test_leave_one_block_out_is_the_refit_without_the_block():
    """`fit.linear.loo.loo_terms` against refits: the squared residuals of each block predicted
    from the others (press), g^T Q^-1 g and log det Q of the block leverages; blocks given as
    their own triangular factors (with the part of y outside their columns added) give the same."""
    from ace_jax.fit.linear import loo
    rng = np.random.default_rng(1)
    k, q, nb, m = 6, 3, 5, 9
    P, Y = rng.normal(size=(nb, m, k)), rng.normal(size=(nb, m, q))
    lin = np.linspace(0.2, 1.5, k)
    A, y = P.reshape(-1, k), Y.reshape(-1, q)
    Qa, Ra = np.linalg.qr(A)
    fac = ls.Factor(jnp.asarray(Ra), jnp.asarray(Qa.T @ y), jnp.asarray((y ** 2).sum(0)), jnp.asarray(float(len(y))), jnp.zeros(()))
    quad, hld, press, mean = (np.asarray(v) for v in loo.loo_terms(jnp.asarray(lin), fac, jnp.asarray(P), jnp.asarray(Y)))
    S = A.T @ A + np.diag(lin)
    assert np.allclose(mean, np.linalg.solve(S, A.T @ y), atol=1e-12)
    pr, qd, ld = 0.0, 0.0, 0.0
    for b in range(nb):
        keep = np.arange(nb) != b
        Ab, yb = P[keep].reshape(-1, k), Y[keep].reshape(-1, q)
        e = Y[b] - P[b] @ np.linalg.solve(Ab.T @ Ab + np.diag(lin), Ab.T @ yb)          # the block, predicted without it
        Qb = np.eye(m) - P[b] @ np.linalg.solve(S, P[b].T)
        pr, qd, ld = pr + (e ** 2).sum(0), qd + np.einsum("iq,ij,jq->q", e, Qb, e), ld + np.linalg.slogdet(Qb)[1]
    assert np.allclose(press, pr, rtol=1e-10) and np.allclose(quad, qd, rtol=1e-10) and np.isclose(2 * hld, ld, rtol=1e-10)
    Rb, cb, eb = [], [], 0.0                                                             # each block as its factor
    for b in range(nb):
        Qf, R = np.linalg.qr(P[b])
        c = Qf.T @ Y[b]
        Rb.append(R); cb.append(c); eb = eb + (Y[b] ** 2).sum(0) - (c ** 2).sum(0)
    quad2, hld2, press2, _ = (np.asarray(v) for v in loo.loo_terms(jnp.asarray(lin), fac, jnp.asarray(np.stack(Rb)), jnp.asarray(np.stack(cb))))
    assert np.allclose(press2 + eb, press, rtol=1e-10) and np.allclose(quad2 + eb, quad, rtol=1e-10) and np.isclose(hld2, hld, rtol=1e-10)
