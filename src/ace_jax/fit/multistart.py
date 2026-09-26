"""Multi-start bounded L-BFGS for the hyperparameter MAP.

The joint linear+GP log marginal likelihood is multimodal: on Cantor-1k with a
PCA feature map, two L-BFGS runs from the SAME start (device vs host-cache LML,
equal to ~1e-9) diverged after ~10 evaluations through round-off and ended 620
nats apart.  The MAP is therefore the best of several starts: the given point
plus seeded draws from the hyperprior, each clipped to the box.

`vg(x) -> (value, grad)` is the log-posterior (maximised) and its gradient, both
as host arrays; a non-finite value or gradient is replaced by a large finite
penalty so the line search backs off instead of aborting (an ill-conditioned
trial point, e.g. sigma_E -> 0).
"""
import numpy as np

_PENALTY = 1e12


def lbfgs_map(vg, x0, lo, hi, maxiter, maxfun=None, log=None):
    """One bounded L-BFGS-B ascent of vg from x0 (clipped to [lo, hi]).
    Returns {x, value, message, nfev, trace} with value = the log-posterior at x
    (-inf if it was never finite)."""
    from scipy.optimize import minimize
    trace = []

    def fg(x):
        v, g = vg(x)
        v, g = float(v), np.asarray(g, float)
        trace.append(v)
        if log is not None:
            log(len(trace), v)
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            return _PENALTY, np.zeros_like(x)
        return -v, -g

    x0 = np.clip(np.asarray(x0, float), lo, hi)
    res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                   options={"maxiter": int(maxiter), "maxfun": int(maxfun or 4 * maxiter)})
    value = -float(res.fun) if float(res.fun) < _PENALTY else -np.inf
    return {"x": np.asarray(res.x, float), "value": value, "message": str(res.message),
            "nfev": int(res.nfev), "trace": trace}


def prior_starts(prior, n, lo, hi, x0, seed=0):
    """n starting points: x0 first, then n-1 draws mu + sigma z from the (log-space
    Gaussian) hyperprior, clipped to [lo, hi].  Coordinates pinned by lo == hi keep
    that value.  Deterministic in seed."""
    if n < 1:
        raise ValueError(f"need at least one start, got {n}")
    mu, sigma = np.asarray(prior.mu, float), np.asarray(prior.sigma, float)
    rng = np.random.default_rng(seed)
    starts = [np.clip(np.asarray(x0, float), lo, hi)]
    for _ in range(n - 1):
        starts.append(np.clip(mu + sigma * rng.standard_normal(mu.shape), lo, hi))
    return starts


def multistart_map(vg, starts, lo, hi, maxiter, maxfun=None, log=None):
    """Run `lbfgs_map` from every start; return (best run, all runs).  Each run
    carries its `start` index; `log(start, eval, value)` if given."""
    runs = []
    for k, x0 in enumerate(starts):
        cb = None if log is None else (lambda i, v, k=k: log(k, i, v))
        r = lbfgs_map(vg, x0, lo, hi, maxiter, maxfun=maxfun, log=cb)
        r["start"] = k
        r["x0"] = np.asarray(x0, float)
        runs.append(r)
    best = max(runs, key=lambda r: r["value"])
    return best, runs
