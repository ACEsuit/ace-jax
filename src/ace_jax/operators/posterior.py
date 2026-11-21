"""The posterior of a block fit's coefficients, from the statistics the fit returns
(`fit_onsite(..., stats=True)`, `fit_offsite`: {design: `fit.DesignStats`}).

One design (a species [pair], L, parity[, same shell]) holds the triangular factor R of its rows;
every right-hand side q (a shell pair and a channel) has its own noise sigma_q and, with
`sigma_c="evidence"`, "loo" or "press", its own prior scale.  Its coefficients are then
    N(mean_q, sigma_q^2 (R^T R + diag(lin_q))^-1),
lin_q the prior precision at unit noise the fit solved with: gamma_j^2 sigma_q^2 / sigma_c,q^2 on
the columns with a smoothness gamma_j and the ridge on the others; for a fit at a fixed prior,
the ridge plus what was given, and sigma_q the residual of the fit, sqrt(RSS / (rows - columns)).

The uncertainty of a prediction has two parts: this one (what more data would remove) and the
noise sigma_q itself, the scatter of the data about the model (what it would not: the model's
own limit and the noise of the reference).  `noise` returns the second per coefficient key."""
import numpy as np
from scipy.linalg import solve_triangular


def _design(st):
    """sigma (n_rhs,) and [(right-hand sides, U)] with U^T U = R^T R + diag(lin) for each distinct prior."""
    R, c = np.asarray(st.factor.R, float), np.asarray(st.factor.c, float)
    k, n = R.shape[1], float(st.factor.n)
    if st.hyper is not None:
        sigma = np.asarray(st.hyper[:, 0], float)
        rho = np.round(2.0 * np.log10(sigma / np.asarray(st.hyper[:, 3], float)), 6)       # log10 sigma^2 / sigma_c^2 as solved with
        lins = {r: np.where(st.gamma > 0, st.gamma ** 2 * 10.0 ** r, st.ridge) for r in np.unique(rho)}
    else:
        rho = np.zeros(c.shape[1])
        lins = {0.0: np.full(k, st.ridge) + (0.0 if st.gamma is None else st.gamma)}
        sigma = None
    groups = []
    for r, lin in lins.items():
        q = np.flatnonzero(rho == r)
        U = np.linalg.qr(np.concatenate([R, np.diag(np.sqrt(lin))]), mode="r")
        groups.append((q, U, lin))
    if sigma is None:                                    # residual of the solve: ||R mu - c||^2 + the part of y outside the columns
        sigma = np.zeros(c.shape[1])
        for q, U, _lin in groups:
            mu = solve_triangular(U, solve_triangular(U, R.T @ c[:, q], trans="T"))
            rss = ((R @ mu - c[:, q]) ** 2).sum(0) + np.maximum(np.asarray(st.factor.yy, float)[q] - (c[:, q] ** 2).sum(0), 0.0)
            sigma[q] = np.sqrt(rss / max(n - k, 1.0))
    return sigma, groups


def noise(stats, nc):
    """{coefficient key: (nc,) the noise sigma of each channel}: the scatter of an L part of that
    shell pair about the model, in the data's unit."""
    out = {}
    for st in stats.values():
        sigma, _ = _design(st)
        for i, key in enumerate(st.keys):
            out[key] = sigma[i * nc:(i + 1) * nc]
    return out


def draws(coef, stats, n, nc, seed=0):
    """{key: (nc (1 + n), columns)}: the fitted coefficients (the first nc rows, as `coef`) and n
    draws from their posterior, draw j's channels in rows nc (1 + j) to nc (2 + j): the
    coefficients of a model with channels (c, c#1, ..., c#n), which predicts the mean and every
    draw's blocks in one call.  Keys without statistics (no columns) are passed through."""
    rng = np.random.default_rng(seed)
    out = {key: np.concatenate([np.asarray(c, float)] * (1 + n)) for key, c in coef.items()}
    for d in sorted(stats, key=str):
        st = stats[d]
        sigma, groups = _design(st)
        k = np.asarray(st.factor.R).shape[1]
        for q, U, _lin in groups:
            dev = solve_triangular(U, rng.normal(size=(k, len(q) * n))).reshape(k, len(q), n) * sigma[q][None, :, None]
            for a, qi in enumerate(q):
                key, ch = st.keys[qi // nc], qi % nc
                for j in range(n):
                    out[key][nc * (1 + j) + ch] += dev[:, a, j]
    return out


def channel_names(channels, n):
    """The channels of a model holding `draws`' coefficients."""
    return tuple(channels) + tuple(f"{c}#{j + 1}" for j in range(n) for c in channels)
