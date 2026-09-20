"""Stable direct solves for the ACE fit (linear M = 0 and hybrid GP M > 0), as
alternatives to the normal-equations Cholesky in objective.posterior.

The Cholesky path forms the Gram G = Phi^T W Phi and factors G + Lambda; that
squares the condition number (kappa(G) = kappa(Phi)^2), so an ill-conditioned
basis needs the smoothness prior Lambda to stay solvable.  The solves here work
on the (weighted, prior-augmented) design matrix directly, at kappa(Phi) not its
square -- ACEfit's QR/RRQR route:

    minimise || Phi_tilde theta - y_tilde ||^2,
    Phi_tilde = [ (w/sigma_t) [Phi_lin | Phi_res] ; R0 ],  y_tilde = [ (w/sigma_t) y ; 0 ]

with the prior square root R0 = blockdiag(diag(gamma/sigma_c), chol(K_MM)^T) (the
residual block is empty when M = 0).  Then Phi_tilde^T Phi_tilde = G + Lambda and
the least-squares solution is the same posterior mean [c ; w] -- but obtained at
kappa(Phi), not kappa(Phi)^2.  The GP joint Gram is much stiffer than the linear
one (the residual features are near-collinear with the linear ACE block), so QR
matters more for M > 0.  Entry points:

  solve_qr           -- dense QR (lineax) on the design streamed into memory
                        once.  Stable, but O(n_obs * L) memory; for small fits.
  solve_qr_streaming -- updating (tall-skinny) QR: keeps only the L x L factor,
                        streams the rows.  Stable AND O(L^2) memory (basis-, not
                        observation-limited) in one pass -- the big-data stable
                        solver.  ~n_batches x the flops of the Cholesky.
  solve_lsqr  -- matrix-free LSQR; matvec/rmatvec stream the rows, nothing of
                 size n_obs * L is ever formed.  O(L) memory -- the option for
                 datasets too large for the design, but it needs O(rank)
                 iterations, each a full streaming pass, so it converges quickly
                 only when the prior conditions the system; for an
                 ill-conditioned ACE Gram prefer solve_qr (design in memory) or
                 the Cholesky fast path.
"""
import jax
import jax.numpy as jnp
import lineax as lx
from jax.scipy.linalg import solve_triangular

from .rows import batch_rows
from .kernels import K_MM


def _row_weights(theta, batch):
    """Per-observation weight w/sigma_t (E, F x3, V x6), flattened like the rows."""
    se, sf, sv = (jnp.exp(theta.log_sigma_E), jnp.exp(theta.log_sigma_F), jnp.exp(theta.log_sigma_V))
    return (batch.w_E / se,
            (jnp.repeat(batch.w_F, 3) / sf),
            (jnp.repeat(batch.w_V, 6) / sv))


def _weighted_rows(prob, theta, batch):
    """Weighted joint [linear | residual] design rows and targets for one batch,
    stacked E|F|V.  Width Dt = len_basis + M (M = 0 gives the linear design)."""
    r = batch_rows(theta, prob.spec, prob.model, prob.ind, prob.cfg, batch)
    Dt = r.E.shape[-1]
    wE, wF, wV = _row_weights(theta, batch)
    Phi = jnp.concatenate([r.E, r.F.reshape(-1, Dt), r.V.reshape(-1, Dt)], 0)
    w = jnp.concatenate([wE, wF, wV])
    y = jnp.concatenate([batch.y_E, batch.y_F.reshape(-1), batch.y_V.reshape(-1)])
    return Phi * w[:, None], y * w                       # (nb, Dt), (nb,)


def _prior_block(prob, theta):
    """The upper-triangular square root R0 of the prior precision Lambda and its
    zero targets, shape (Dt, Dt) with Dt = len_basis + M.  Lambda =
    blockdiag(diag(gamma^2/sigma_c^2), K_MM), so R0 = blockdiag(diag(gamma/sigma_c),
    chol(K_MM)^T) -- upper triangular (both blocks are), so it also seeds the
    streaming QR.  M = 0 recovers diag(gamma/sigma_c)."""
    L, M = prob.cfg.len_basis, prob.ind.XM.shape[0]
    sc = jnp.exp(theta.log_sigma_c)
    R0 = jnp.zeros((L + M, L + M)).at[jnp.arange(L), jnp.arange(L)].set(prob.gamma / sc)
    if M > 0:
        LMM = jnp.linalg.cholesky(K_MM(theta, prob.spec, prob.ind.XM, prob.ind.SM, prob.ind.ZM, prob.ind.embed))
        R0 = R0.at[L:, L:].set(LMM.T)                    # chol(K_MM)^T, upper triangular
    return R0, jnp.zeros(L + M)


def stacked_design(prob, ds, theta):
    """Materialise the full weighted, prior-augmented joint design (one streaming
    pass).  Returns Phi_tilde (n_obs + Dt, Dt) and y_tilde, Dt = len_basis + M."""
    rows = [_weighted_rows(prob, theta, jax.tree.map(lambda a: a[b], ds))
            for b in range(ds.n_batches)]
    Pp, yp = _prior_block(prob, theta)
    Phi = jnp.concatenate([P for P, _ in rows] + [Pp], 0)
    y = jnp.concatenate([yv for _, yv in rows] + [yp])
    return Phi, y


def solve_qr(prob, ds, theta):
    """Dense QR least-squares on the stacked joint design (lineax).  Returns the
    posterior mean [c ; w] (theta_MAP), stably (kappa, not kappa^2).  M >= 0."""
    Phi, y = stacked_design(prob, ds, theta)
    op = lx.MatrixLinearOperator(Phi)
    return lx.linear_solve(op, y, solver=lx.QR()).value


def solve_qr_streaming(prob, ds, theta):
    """Updating (tall-skinny) QR: stream the weighted rows, maintaining only the
    L x L triangular factor R and the transformed RHS d = Q^T y_tilde.  Memory is
    O(L^2) -- basis-limited like the Gram, NOT observation-limited like the dense
    solve_qr -- but the factorisation sees kappa(Phi), not kappa(Phi)^2 like the
    Cholesky.  One streaming pass; nothing of size n_obs x Dt is formed.  Seeded
    with the prior square root R0 = blockdiag(diag(gamma/sigma_c), chol(K_MM)^T)
    (upper triangular), so the prior enters exactly once.  Dt = len_basis + M, so
    this covers the GP (M > 0): the residual weights w couple through K_MM.

    Per batch it re-triangularises [R ; A_k] (a Dt^3-scale QR), so it is
    ~n_batches times the flops of the Cholesky's single factorisation -- the price
    of the kappa (not kappa^2) conditioning at O(Dt^2) memory."""
    R0, d0 = _prior_block(prob, theta)                       # (Dt, Dt) upper tri, zeros(Dt)

    def body(carry, batch):
        R, d = carry
        Aw, yw = _weighted_rows(prob, theta, batch)          # (nb, Dt), (nb,)
        Q, R2 = jnp.linalg.qr(jnp.concatenate([R, Aw], 0), mode="reduced")   # (Dt+nb, Dt), (Dt, Dt)
        d2 = Q.T @ jnp.concatenate([d, yw])
        return (R2, d2), None

    (R, d), _ = jax.lax.scan(body, (R0, d0), ds)
    return solve_triangular(R, d, lower=False)                # back-substitution R theta = d


# ---------------------------------------------------------------------------
# Matrix-free LSQR (Paige & Saunders 1982), damping 0 (the prior is already a
# block of Phi_tilde).  A fixed number of Golub-Kahan steps as a scan; the
# operator is any (matvec, rmatvec) pair, so the design is never materialised.
# Note: LSQR needs O(rank) iterations, each a full streaming pass, so it suits
# the ill-conditioned / memory-bound regime, not the moderate-L fast path
# (that is Cholesky-on-Gram); it converges here because the prior regularises.

def lsqr(matvec, rmatvec, b, n, maxiter=200):
    """Least-squares min ||A x - b|| via LSQR.  matvec: R^n -> R^m (A v),
    rmatvec: R^m -> R^n (A^T u).  Returns x in R^n."""
    beta = jnp.linalg.norm(b)
    u = b / jnp.where(beta > 0, beta, 1.0)
    v0 = rmatvec(u)
    alpha = jnp.linalg.norm(v0)
    v = v0 / jnp.where(alpha > 0, alpha, 1.0)
    w = v
    x = jnp.zeros(n)
    phibar, rhobar = beta, alpha

    def step(carry, _):
        u, v, w, x, phibar, rhobar, alpha = carry
        u2 = matvec(v) - alpha * u
        beta = jnp.linalg.norm(u2)
        u2 = u2 / jnp.where(beta > 0, beta, 1.0)
        v2 = rmatvec(u2) - beta * v
        alpha2 = jnp.linalg.norm(v2)
        v2 = v2 / jnp.where(alpha2 > 0, alpha2, 1.0)
        rho = jnp.sqrt(rhobar ** 2 + beta ** 2)
        c, sn = rhobar / rho, beta / rho
        theta = sn * alpha2
        rhobar2 = -c * alpha2
        phi = c * phibar
        phibar2 = sn * phibar
        x = x + (phi / rho) * w
        w = v2 - (theta / rho) * w
        return (u2, v2, w, x, phibar2, rhobar2, alpha2), None

    return jax.lax.scan(step, (u, v, w, x, phibar, rhobar, alpha), None, length=maxiter)[0][3]


def streamed_operators(prob, ds, theta):
    """The matrix-free (matvec, rmatvec, y_tilde) of the weighted, prior-augmented
    design -- the same linear operator stacked_design materialises, but streamed:
    matvec(x) = Phi_tilde x, rmatvec(u) = Phi_tilde^T u, nothing of size n_obs*Dt
    is formed.  Dt = len_basis + M (covers the GP)."""
    Dt = prob.cfg.len_basis + prob.ind.XM.shape[0]
    R0 = _prior_block(prob, theta)[0]                           # (Dt, Dt) prior sqrt
    batches = [jax.tree.map(lambda a: a[b], ds) for b in range(ds.n_batches)]

    def _rows(batch):
        r = batch_rows(theta, prob.spec, prob.model, prob.ind, prob.cfg, batch)
        return r.E, r.F.reshape(-1, Dt), r.V.reshape(-1, Dt)

    def matvec(x):
        parts = []
        for batch in batches:
            rE, rF, rV = _rows(batch)
            wE, wF, wV = _row_weights(theta, batch)
            parts += [rE @ x * wE, (rF @ x) * wF, (rV @ x) * wV]
        parts.append(R0 @ x)
        return jnp.concatenate(parts)

    def rmatvec(u):
        acc = jnp.zeros(Dt)
        off = 0
        for batch in batches:
            rE, rF, rV = _rows(batch)
            wE, wF, wV = _row_weights(theta, batch)
            nE, nF, nV = wE.shape[0], wF.shape[0], wV.shape[0]
            uE, uF, uV = u[off:off + nE], u[off + nE:off + nE + nF], u[off + nE + nF:off + nE + nF + nV]
            off += nE + nF + nV
            acc = acc + rE.T @ (wE * uE) + rF.T @ (wF * uF) + rV.T @ (wV * uV)
        return acc + R0.T @ u[off:off + Dt]

    y = jnp.concatenate([_weighted_rows(prob, theta, b)[1] for b in batches] + [jnp.zeros(Dt)])
    return matvec, rmatvec, y


def solve_lsqr(prob, ds, theta, maxiter=None):
    """Matrix-free LSQR on the streamed joint design.  Nothing of size n_obs * Dt
    is materialised.  Converges quickly only when the prior conditions the system
    (see the module docstring); for an ill-conditioned Gram prefer solve_qr /
    solve_qr_streaming or the Cholesky fast path.  Dt = len_basis + M."""
    Dt = prob.cfg.len_basis + prob.ind.XM.shape[0]
    matvec, rmatvec, y = streamed_operators(prob, ds, theta)
    return lsqr(matvec, rmatvec, y, Dt, maxiter=maxiter if maxiter is not None else 2 * Dt)
