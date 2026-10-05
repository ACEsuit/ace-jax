"""Projected Newton polish of a bounded smooth maximum (an evidence or a log-posterior).

Shared by the ARD evidence fit (`ard.fit_ard`) and the hyperparameter MAP (`pipeline.mapfit`):
L-BFGS-B stops at a roundoff-dependent point (a line-search failure on a noisy objective, or a
loose scaled tolerance), so both polish its endpoint with a Hessian, to tolerances measured from
the objective's own roundoff.  `ev` is any object with `value_and_grad(h) -> (v, g)` and
`hessian(h)`, of the quantity being MAXIMISED, as host arrays."""
import numpy as np


def _projected_gradient(x, g, lo, hi):
    """x - P(x - g) for the minimisation gradient g: zero for a coordinate at a bound that g pushes out."""
    return x - np.clip(x - g, lo, hi)


def _evidence_noise(evaluate, x, F, g, H):
    """Roundoff of F and of each gradient component at x: their spread, after removing the linear
    changes g.dx and H dx, over x and four probes x + k 2^-50 (|x| + 1) (k = +-1, +-2: absolute where
    x = 0); floored at 4 ulp.  Returns (noise_F, noise_g (P,))."""
    s = 2.0 ** -50 * (np.abs(x) + 1.0)
    Fs, gs = [F], [g]
    for k in (1.0, -1.0, 2.0, -2.0):
        Fk, gk = evaluate(x + k * s)
        Fs.append(Fk - k * float(g @ s))
        gs.append(gk - k * (H @ s))
    gs = np.array(gs)
    return (max(float(np.ptp(Fs)), 4 * float(np.spacing(abs(F)))),
            np.maximum(np.ptp(gs, axis=0), 4 * np.spacing(np.abs(gs).max(axis=0))))


def _box_qp(H, g, l, u):
    """argmin_d g.d + 1/2 d^T H d over l <= d <= u (H positive definite, l <= 0 <= u) by the primal
    active-set method from d = 0: solve on the free coordinates, stop at the first bound the step
    meets (it joins the working set), release the working-set coordinate whose multiplier has the
    wrong sign (the largest) once the free solve is reached.  Finite for PD H; P is ~6."""
    P = len(g)
    d = np.zeros(P)
    W = ((l == 0) & (g > 0)) | ((u == 0) & (g < 0))
    for _ in range(20 * P + 20):
        fr = ~W
        p = np.zeros(P)
        if fr.any():
            r = g + H @ d
            p[fr] = -np.linalg.solve(H[np.ix_(fr, fr)], r[fr])
        alpha, block = 1.0, -1
        for i in np.flatnonzero(fr):
            if p[i] < 0 and d[i] + p[i] < l[i]:
                a = (l[i] - d[i]) / p[i]
            elif p[i] > 0 and d[i] + p[i] > u[i]:
                a = (u[i] - d[i]) / p[i]
            else:
                continue
            if a < alpha:
                alpha, block = a, i
        d = d + alpha * p
        if block >= 0:
            d[block] = l[block] if p[block] < 0 else u[block]
            W[block] = True
            continue
        r = g + H @ d                                    # the free-face minimum: check the multipliers
        wrong = W & (((d == l) & (r < 0)) | ((d == u) & (r > 0)) | ((d != l) & (d != u)))
        if not wrong.any():
            return d
        W[np.flatnonzero(wrong)[np.argmax(np.abs(r[wrong]))]] = False
    return d


def _newton_step(H, g, x, lo, hi):
    """The projected Newton step: the exact box-constrained minimiser (`_box_qp`) of the quadratic
    model g.d + 1/2 d^T Hm d over x + d in [lo, hi], Hm = H with |eigenvalues| floored at 1e-10 of the
    largest (positive definite).  A coordinate the unconstrained step would carry out of the box ends
    on its bound with the rest re-optimised, and one at a bound whose model gradient pushes outward
    stays there.  Returns (d, decrement^2 = -2 x the model's predicted change of F, >= 0)."""
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    Hm = (V * np.maximum(np.abs(w), 1e-10 * max(np.abs(w).max(), 1e-300))) @ V.T
    d = _box_qp(Hm, g, lo - x, hi - x)
    return d, max(0.0, -2.0 * float(g @ d + 0.5 * d @ Hm @ d))


def newton_polish(ev, x, lo, hi, maxiter=50, gradient_floor=False):
    """Converge a bounded evidence maximum by projected Newton with the exact Hessian (ev.hessian).

    Minimises F = -log p(D|h) over the box [lo, hi] from x (an L-BFGS-B endpoint).  Each step is the
    box-constrained minimiser of the Newton model (`_newton_step`: Hessian modified to positive
    definite; the active set -- bound coordinates the model pushes outward, and those the step would
    carry past a bound -- decided by an exact active-set QP), so it stays in the box and decreases
    the model.  Every tolerance is the measured roundoff at x (`_evidence_noise`), so none depends
    on F's additive constants (sum_q n_q log sigma_q, 1/2 sum_q yy_q / sigma_q^2):

    - converged when every projected-gradient component is within 10x its own roundoff -- tested
      only after a first step if the start's predicted decrease is resolvable (decrement^2 > 10x F's
      roundoff), so a resolvably improvable L-BFGS endpoint is always polished -- (the
      gradient stays resolved far below F's noise: cond(S) ~ 5e13 leaves F to ~1e-5 nats, the
      gradient to ~1e-5 absolute but the step to ~1e-6);
    - a step is accepted on an Armijo decrease of F (backtracked) while the model's predicted
      decrease (decrement^2 / 2) exceeds 10x F's roundoff, and otherwise -- or failing that -- at full
      length if it halves ||pg||_inf without raising F by more than 10x its roundoff;
    - with no step accepted it stops: converged if the decrement^2 is within 10x F's roundoff (no
      decrease is resolvable), else not; not converged after maxiter.

    If the endpoint's F is worse than the start's by more than 10x F's roundoff the start is kept
    (not converged) and reported (pg, decrement, and its roundoff `noise` / `gnoise`, re-measured
    whenever the returned point is not where they were last measured, so info["hessian"], ev.hessian
    at the returned point, comes with them).  Cost per iteration: one Hessian, four noise
    probes, one to a few evaluations.

    gradient_floor (the MAP polish; False keeps the ARD evidence fit bit-identical): F's roundoff
    does not end the polish while a projected-gradient component exceeds its own roundoff gnoise.
    Then an improving Newton step is always tried (Armijo, else the longest of t = 1, 1/2, ... that
    lowers ||pg||_inf without raising F past its roundoff), and a stop counts as converged only
    when every component is within 10x its gnoise.  On a weakly curved direction (12 Si configs:
    log sigma_c, |pg| 14.8 against gnoise 0.15) the decrement can sit below F's roundoff while the
    gradient is resolvably nonzero; without this the polish stopped there, 2.9 nats short.  Deterministic: a fixed sequence of compiled evaluations and LAPACK calls on
    P x P matrices, no randomness."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    x = np.clip(np.asarray(x, float), lo, hi)

    def evaluate(z):
        v, g = ev.value_and_grad(z)
        return -v, -g

    F, g = evaluate(x)
    if not (np.isfinite(F) and np.all(np.isfinite(g))):
        return x, {"converged": False, "message": "non-finite evidence at the L-BFGS endpoint", "steps": 0,
                   "hessian_evals": 0, "pg_start": float("nan"), "pg": float("nan"), "decrement": float("nan"),
                   "noise": float("nan"), "gnoise": [float("nan")] * len(x), "hessian": None}
    x_start, F_start = x, F
    pgv = _projected_gradient(x, g, lo, hi)
    pg = pg0 = float(np.abs(pgv).max())
    steps, n_hess, converged, dec, dec0, noise = 0, 0, False, float("nan"), float("nan"), 0.0
    gnoise, noise_at = None, None
    message = f"maxiter {maxiter}"
    for _ in range(maxiter):
        H = -ev.hessian(x)
        n_hess += 1
        noise, gnoise = _evidence_noise(evaluate, x, F, g, H)
        noise_at = x
        d, dec = _newton_step(H, g, x, lo, hi)
        if steps == 0 and n_hess == 1:
            dec0 = dec
        if not np.any(d):
            converged = True
            message = f"converged (zero projected Newton step; |pg| {pg:.2e})"
            break
        # the gradient test only after a first step whenever that step's decrease is resolvable: the
        # L-BFGS endpoint (BLAS/layout-dependent) is never returned unpolished if it can be improved
        if (steps > 0 or dec <= 10 * noise) and np.all(np.abs(pgv) <= 10 * gnoise):
            converged = True
            message = f"converged (|pg| {pg:.2e} within 10x the gradient roundoff {gnoise.max():.1e})"
            break
        accepted = False
        g_resolved = gradient_floor and bool(np.any(np.abs(pgv) > gnoise))
        if dec > 10 * noise or g_resolved:               # a resolved decrease: backtrack to Armijo
            for t in 0.5 ** np.arange(40):
                xn = np.clip(x + t * d, lo, hi)
                if np.array_equal(xn, x):
                    break
                Fn, gn = evaluate(xn)
                if np.isfinite(Fn) and np.all(np.isfinite(gn)) and Fn <= F + 1e-4 * float(g @ (xn - x)):
                    accepted = True
                    break
        if not accepted:                                 # the full step, if it halves ||pg|| within F's noise
            xn = np.clip(x + d, lo, hi)
            Fn, gn = evaluate(xn)
            accepted = bool(np.isfinite(Fn) and np.all(np.isfinite(gn)) and Fn <= F + 10 * noise
                            and np.abs(_projected_gradient(xn, gn, lo, hi)).max() <= 0.5 * pg)
        if not accepted and g_resolved:                  # the gradient is resolved: any step that lowers it
            for t in 0.5 ** np.arange(20):
                xn = np.clip(x + t * d, lo, hi)
                Fn, gn = evaluate(xn)
                if (np.isfinite(Fn) and np.all(np.isfinite(gn)) and Fn <= F + 10 * noise
                        and np.abs(_projected_gradient(xn, gn, lo, hi)).max() < pg):
                    accepted = True
                    break
        if not accepted:
            converged = dec <= 10 * noise and (not gradient_floor or bool(np.all(np.abs(pgv) <= 10 * gnoise)))
            message = (f"{'roundoff floor' if converged else 'no acceptable Newton step'} (|pg| {pg:.2e}, gradient "
                       f"roundoff {gnoise.max():.1e}, decrement^2 {dec:.1e}, F roundoff {noise:.1e})")
            break
        x, F, g = xn, Fn, gn
        pgv = _projected_gradient(x, g, lo, hi)
        pg = float(np.abs(pgv).max())
        steps += 1
    if F > F_start + 10 * noise:
        x, F, converged, pg, dec = x_start, F_start, False, pg0, dec0
        message += "; kept the L-BFGS endpoint (lower F)"
    if gnoise is None or not np.array_equal(noise_at, x):   # the reported roundoff is the returned point's
        Fx, gx = evaluate(x)
        H = -ev.hessian(x)
        n_hess += 1
        noise, gnoise = _evidence_noise(evaluate, x, Fx, gx, H)
    return x, {"converged": bool(converged), "message": message, "steps": steps, "hessian_evals": n_hess,
               "pg_start": pg0, "pg": pg, "decrement": dec, "noise": noise, "gnoise": np.asarray(gnoise).tolist(),
               "hessian": -H}       # ev.hessian at the returned x (an array: callers drop it before JSON)
