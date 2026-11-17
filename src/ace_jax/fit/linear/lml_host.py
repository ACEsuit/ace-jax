"""The QR evidence of a diagonal-prior linear model (objective.log_evidence_qr) evaluated on the host with its
structure.  Written for the quantities q of any statistics object (`stats.quantities_of`); a potential fit
has E, F and V.

The evidence reads its hyperparameters only through the scales p = (s, a_q ...) = (1 / sigma_c^2,
1 / sigma_q^2):  S = s M_0 + C + sum_q a_q R_q^T R_q, with M_0 = diag(gamma^2) on the readout columns and
C = diag(e0_prec) on the joint-E0 columns.  The dense form QR-factors the stacked (4L x L) matrix
[diag(sqrt(lin)); sqrt(a_q) R_q] per evaluation, forms Q, and autodiff differentiates through both: ~25 L^3
flops (590 s per L-BFGS step at L = 21,000 on 48 cores).  Every block is diagonal or triangular, so here:

- factor: three triangle-triangle merges by LAPACK tpqrt onto diag(sqrt(lin)) (prior rows first, as in the
  dense form), with the targets sqrt(a_q) c_q as an extra column, so the last diagonal is the residual norm and
  y'y - v'v is never formed; R_S, v, r = the (L+1) factor.  Then W = R_S^-1 (trtri), mu = R_S^-1 v.
- gradient in p:  d logdet S / d p_k = tr(S^-1 M_k) = ||Y_k||_F^2 (Y_q = R_q W, Y_0 = diag(gamma) W_r), and
  d resid / d a_q = ||R_q mu - c_q||^2 + kappa_q (envelope theorem; kappa_q = the residual outside c_q: rho_q^2
  from float32 rows, else yy_q - |c_q|^2), d resid / d s = mu_r' diag(gamma^2) mu_r.
- Hessian in p (exact):  d2 logdet = -<X_k, X_l>_F with X_k = Y_k^T Y_k (syrk), d2 resid = -2 z_k . z_l with
  z_k = W^T u_k, u_q = R_q^T (c_q - R_q mu), u_0 = -gamma^2 mu_r.

Value and gradient: ~3 merges + L^3/3 + 3 trmm.  The Hessian (Newton polish, Laplace) adds 3 trmm + 4 syrk.
JAX sees it through a custom_jvp chain (value -> gradient -> Hessian), so jax.grad, forward-over-reverse HVPs
and jax.hessian all work; vmap evaluates point by point.  Peak memory beyond the statistics ~6 L^2 doubles
(value, gradient) or ~8 (Hessian), against ~20 for the dense form.  CPU only: the statistics are host arrays."""
import numpy as np

import jax
import jax.numpy as jnp

TPQRT_NB = 256


def _tri_merge(A, R, c, w):
    """A <- the (n+1) x (n+1) R factor of [A; w [R, c]] (tpqrt, both triangles)."""
    import scipy.linalg.lapack as la
    n = R.shape[0]
    B = np.zeros((n + 1, n + 1), order="F")
    B[:n, :n] = R
    B[:n, n] = c
    if w != 1.0:
        B *= w
    a, _, _, info = la.dtpqrt(n + 1, min(TPQRT_NB, n + 1), A, B, overwrite_a=1, overwrite_b=1)
    if info != 0:
        raise np.linalg.LinAlgError(f"dtpqrt info={info}")
    return a


def _frob_sym(X, Y):
    """<X, Y>_F of two symmetric matrices given by their upper triangles (syrk output)."""
    d = np.einsum("ii,ii->", X, Y)
    return 2.0 * np.sum(np.triu(X) * np.triu(Y)) - d


class HostQREvidence:
    """The evidence of statistics `qs` (QR form: `.factors` and `.quantities`, or a sequence of Factor) under
    the prior of `prob` (`.gamma`, and `.e0_prec` for columns with a fixed precision), by the structured host
    evaluation (module docstring).  `lml(theta)` is the JAX entry point: theta has `log_sigma_c` and one
    `log_sigma_<quantity>` per quantity.  n_factor counts factorisations (one per theta: value, gradient and
    Hessian at the same theta share it).

    shared: the caller ties every quantity's noise to one (a potential fit: sigma_E and sigma_V to sigma_F,
    cfg.noise "shared", paramset.tied_likelihood), so only that scale is read (True: the statistics'
    `shared_quantity`, else the first; or the quantity's label) and the factors are merged once, here: one
    tpqrt merge and one trmm per evaluation instead of one per quantity.  The merge's last diagonal r0 joins the residual outside c:
    kappa = sum_q kappa_q + r0^2.  Only valid under that tie: an untied caller gets the wrong derivatives."""

    def __init__(self, qs, prob, shared=False):
        from .stats import quantities_of
        fs, labels = quantities_of(qs)
        R = [np.asarray(f.R, np.float64) for f in fs]
        c = [np.asarray(f.c, np.float64) for f in fs]
        n = np.array([float(f.n) for f in fs])
        self.logw = sum(float(f.logw) for f in fs)
        if fs[0].rho is not None:
            kappa = np.array([float(f.rho) ** 2 for f in fs])
        else:
            kappa = np.array([float(f.yy) - float(cq @ cq) for cq, f in zip(c, fs)])
        one = getattr(qs, "shared_quantity", labels[0]) if shared is True else shared
        self.shared, self.sigmas = bool(shared), (one,) if shared else labels
        if shared:
            L = R[0].shape[0]
            A = np.zeros((L + 1, L + 1), order="F")
            A[:L, :L], A[:L, L] = R[0], c[0]
            for Rq, cq in zip(R[1:], c[1:]):
                A = _tri_merge(A, Rq, cq, 1.0)
            R, c = [np.triu(A[:L, :L])], [A[:L, L].copy()]
            kappa, n = np.array([kappa.sum() + float(A[L, L]) ** 2]), np.array([n.sum()])
            del A
        self.R, self.c, self.kappa, self.n = R, c, kappa, n
        self.g2 = np.asarray(prob.gamma, np.float64) ** 2
        e0p = getattr(prob, "e0_prec", None)
        self.e0p = np.zeros(0) if e0p is None else np.asarray(e0p, np.float64)
        self.Lr, self.L = self.g2.size, self.g2.size + self.e0p.size
        self.loglin0 = float(np.sum(np.log(self.g2)) + np.sum(np.log(self.e0p)))
        self.n_factor = 0
        self._key, self._f = None, None
        self._core = self._build()

    # -- host side ------------------------------------------------------------------------------------------
    def _factor(self, t):
        t = np.asarray(t, np.float64)
        key = t.tobytes()
        if key == self._key:
            return self._f
        import scipy.linalg as sla
        import scipy.linalg.lapack as la
        self.n_factor += 1
        p = np.exp(t)
        s, a = p[0], p[1:]
        n, Lr = self.L, self.Lr
        A = np.zeros((n + 1, n + 1), order="F")
        d = np.arange(n)
        A[d, d] = np.sqrt(np.concatenate([self.g2 * s, self.e0p]))
        for R, c, aq in zip(self.R, self.c, a):
            A = _tri_merge(A, R, c, np.sqrt(aq))
        RS, v, r = np.triu(A[:n, :n]), A[:n, n].copy(), float(A[n, n])
        del A
        mu = sla.solve_triangular(RS, v, lower=False)
        W, info = la.dtrtri(RS, lower=0)
        if info != 0:
            raise np.linalg.LinAlgError(f"dtrtri info={info}")
        W = np.triu(W)
        K = len(self.R)
        tr = np.empty(K + 1)
        Wr = W[:Lr]
        tr[0] = float(np.einsum("i,ij,ij->", self.g2, Wr, Wr))
        rq = np.empty(K)
        for k, (R, c) in enumerate(zip(self.R, self.c)):
            Yt = self._Yt(W, R)
            tr[k + 1] = float(np.einsum("ij,ij->", Yt, Yt))
            e = R @ mu - c
            rq[k] = float(e @ e) + self.kappa[k]
            del Yt
        resid = r * r + float(a @ self.kappa)
        logdet = 2.0 * float(np.sum(np.log(np.abs(np.diag(RS)))))
        loglin = self.Lr * t[0] + self.loglin0
        logtau = self.logw + float(self.n @ t[1:])
        N = float(self.n.sum())
        val = -0.5 * resid - 0.5 * (logdet - loglin - logtau) - 0.5 * N * np.log(2.0 * np.pi)
        dres = np.concatenate([[float(self.g2 @ mu[:Lr] ** 2)], rq])
        cnt = np.concatenate([[float(self.Lr)], self.n])
        gp = -0.5 * dres - 0.5 * tr + 0.5 * cnt / p
        self._key, self._f = key, dict(p=p, gp=gp, val=val, W=W, mu=mu, cnt=cnt)
        return self._f

    @staticmethod
    def _Yt(W, R):
        """(R W)^T = W^T R^T (L x L), R upper triangular: trmm with W, R^T read in place (R is C-ordered)."""
        import scipy.linalg.blas as blas
        return blas.dtrmm(1.0, W, np.asfortranarray(R.T), side=0, lower=0, trans_a=1)

    def value(self, t):
        return np.asarray(self._factor(t)["val"], np.float64)

    def grad(self, t):
        f = self._factor(t)
        return np.asarray(f["p"] * f["gp"], np.float64)

    def hess(self, t):
        import scipy.linalg.blas as blas
        f = self._factor(t)
        p, gp, W, mu = f["p"], f["gp"], f["W"], f["mu"]
        Lr = self.Lr
        u0 = np.zeros(self.L)
        u0[:Lr] = -self.g2 * mu[:Lr]
        us = [u0] + [R.T @ (c - R @ mu) for R, c in zip(self.R, self.c)]
        Z = np.stack([W.T @ u for u in us])
        Hres = -2.0 * Z @ Z.T
        Wg = np.asfortranarray(np.sqrt(self.g2)[:, None] * W[:Lr])
        X = [blas.dsyrk(1.0, Wg, trans=1)]
        del Wg
        for R in self.R:
            Yt = self._Yt(W, R)
            X.append(blas.dsyrk(1.0, Yt, trans=0))
            del Yt
        Hld = -np.array([[_frob_sym(X[k], X[m]) for m in range(len(X))] for k in range(len(X))])
        del X
        Hp = -0.5 * Hres - 0.5 * Hld - 0.5 * np.diag(f["cnt"] / p ** 2)
        return np.asarray(p[:, None] * Hp * p[None, :] + np.diag(p * gp), np.float64)

    # -- JAX side -------------------------------------------------------------------------------------------
    def _build(self):
        f64 = jnp.float64

        def cb(fn, shape):
            return lambda t: jax.pure_callback(fn, jax.ShapeDtypeStruct(shape, f64), t, vmap_method="sequential")
        m = 1 + len(self.sigmas)
        hess = cb(self.hess, (m, m))

        @jax.custom_jvp
        def grad(t):
            return cb(self.grad, (m,))(t)

        @grad.defjvp
        def _grad_jvp(primals, tangents):
            (t,), (dt,) = primals, tangents
            return grad(t), hess(t) @ dt

        @jax.custom_jvp
        def core(t):
            return cb(self.value, ())(t)

        @core.defjvp
        def _core_jvp(primals, tangents):
            (t,), (dt,) = primals, tangents
            return core(t), grad(t) @ dt
        return core

    def lml(self, theta):
        """The evidence at Hypers theta (a JAX function of theta; only sigma_c and the noises are read)."""
        t = -2.0 * jnp.stack([jnp.asarray(getattr(theta, f"log_sigma_{q}"), jnp.float64)
                              for q in ("c",) + self.sigmas])
        return self._core(t)
