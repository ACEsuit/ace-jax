"""Joint linear + residual-GP marginal likelihood with the linear design rows
cached in host RAM, so a likelihood evaluation never touches the ACE basis.

Why.  make_lml(cache_linear=True) caches the theta-independent linear Gram G_BB,
but the cross block G_BM = Phi_B^T Phi_M(theta) still needs the linear rows
Phi_B, which it recomputes -- the full ACE descriptor Jacobian -- on every
evaluation (twice, counting the checkpointed reverse pass).  At the production
Cantor basis (L ~ 2.8e4, 3200 configs) that is ~80 min per evaluation.

How.  The residual GP sees only d projected channels U0 = X @ Pmap, and those
inputs and their edge Jacobian JU0 = Pmap^T J are theta-independent.  With a
pair-density map (--density pair) they come from the pair radials alone
(rows.pair_feature_inputs) and are recomputed each evaluation; with a low-rank
PCA map of the full descriptor (--density pca) they are computed in the one ACE
pass and cached in host RAM next to the linear rows (d * Ncap * K * 3 * 8 B per
batch).  The weighted linear rows w * Phi_B of every batch are
computed ONCE (the same pass accumulates the linear statistics) and kept on the
host, compacted per quantity to the live rows; each evaluation streams them back
in chunks for G_BM.  The gradient streams twice: forward for the residual
statistics, then each chunk's VJP against the LML's cotangent for them.

Memory: host ~ rows * L * 8 B (~107 GB for Cantor-3200 at L = 27.6k), plus the
cached PCA inputs (~0.45 GB * d at Cantor-3200); device as make_lml minus the
per-evaluation ACE Jacobian, plus one chunk of rows.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .hypers import from_array
from .objective import log_marginal_likelihood
from .rows import linear_rows, pair_feature_inputs, residual_rows_from_inputs
from .stats import ResidualStats, Stats, assemble_statistics, batch_linear_stats

_Q = ("E", "F", "V")


def _weights(batch):
    return (batch.w_E, jnp.repeat(batch.w_F, 3), jnp.repeat(batch.w_V, 6))


def _targets(batch):
    return (batch.y_E, batch.y_F.reshape(-1), batch.y_V.reshape(-1))


def _flat(R):
    """Rows (E (C,D), F (Ncap,3,D), V (C,6,D)) -> per-quantity row matrices."""
    D = R.E.shape[-1]
    return (R.E, R.F.reshape(-1, D), R.V.reshape(-1, D))


class HostCachedLML:
    """LML(a) and value_and_grad(a) for the joint linear+GP model, a the log
    hyperparameter vector (to_array order).  Equals make_lml(prob, ds) to f64
    roundoff (tests/test_gp_hostcache.py).  `chunk` batches per host transfer."""

    def __init__(self, prob, ds, chunk=64):
        cfg, ind = prob.cfg, prob.ind
        if ind.XM.shape[0] == 0:
            raise ValueError("HostCachedLML is for the GP arm (M > 0); the linear arm's "
                             "likelihood is already cached by make_lml")
        P = np.asarray(ind.Pmap)
        if P.shape[0] != cfg.D:
            raise ValueError(f"Pmap has {P.shape[0]} rows, expected D = {cfg.D}")
        self.pair_only = not np.any(P[:cfg.n_B] != 0)
        if not self.pair_only and P.shape[1] >= cfg.D:
            raise ValueError("HostCachedLML needs a narrow feature map (--density pair or --density pca): "
                             "the isotropic full-descriptor map would cache a D-wide input Jacobian "
                             "per batch, as large as the ACE Jacobian itself")
        self.prob, self.ds, self.chunk = prob, ds, int(chunk)
        self.nb = int(ds.y_E.shape[0])

        # fixed per-quantity row budgets from the weights alone (no ACE needed)
        w_host = [np.asarray(w) for w in jax.vmap(_weights)(ds)]      # each (nb, rows_q)
        self.idx, self.valid = [], []
        for w in w_host:
            live = w > 0
            R = max(1, int(live.sum(1).max()))
            idx = np.zeros((self.nb, R), np.int32)
            val = np.zeros((self.nb, R), bool)
            for b in range(self.nb):
                k = np.flatnonzero(live[b])
                idx[b, :k.size] = k
                val[b, :k.size] = True
            self.idx.append(idx); self.valid.append(val)

        # ONE pass over the ACE basis: linear statistics (device) + weighted rows (host)
        # (+ the PCA map's residual inputs U0, JU0, also theta-independent)
        L = cfg.len_basis
        self.rows = [np.zeros((self.nb, i.shape[1], L)) for i in self.idx]
        Ncap, K = ds.nbr.shape[1:]
        if not self.pair_only:
            d = P.shape[1]
            self.U0 = np.zeros((self.nb, Ncap, d))
            self.JU0 = np.zeros((self.nb, Ncap, K, d, 3))
        Pj = jnp.asarray(P)

        def one_fn(b):
            R, X, J = linear_rows(prob.model, cfg, b)
            Bw = tuple(Rq * wq[:, None] for Rq, wq in zip(_flat(R), _weights(b)))
            inp = () if self.pair_only else (X @ Pj, jnp.einsum("nkDa,Dq->nkqa", J.reshape(Ncap, K, -1, 3), Pj))
            return batch_linear_stats(prob.model, cfg, b), Bw, inp
        one = jax.jit(one_fn)
        lin = None
        for b in range(self.nb):
            st, Bw, inp = one(jax.tree.map(lambda x: x[b], ds))
            lin = st if lin is None else jax.tree.map(jnp.add, lin, st)
            for q in range(3):
                self.rows[q][b] = np.asarray(Bw[q])[self.idx[q][b]] * self.valid[q][b][:, None]
            if not self.pair_only:
                self.U0[b], self.JU0[b] = np.asarray(inp[0]), np.asarray(inp[1])
        self.lin = jax.block_until_ready(lin)

        self._fwd = jax.jit(self._chunk_stats)
        self._bwd = jax.jit(lambda a, xs, ct: jax.vjp(lambda t: self._chunk_stats(t, xs), a)[1](ct)[0])
        self._head = jax.jit(jax.value_and_grad(self._head_fn, argnums=(0, 1)))
        self._head_value = jax.jit(self._head_fn)

    # ------------------------------------------------------------------ pieces
    def _batch_stats(self, theta, batch, Bw, idx, valid, inp):
        prob = self.prob
        U0, JU0 = (pair_feature_inputs(prob.model, prob.ind, prob.cfg, batch) if self.pair_only else inp)
        Mrows = _flat(residual_rows_from_inputs(theta, prob.spec, prob.ind, prob.cfg, batch, U0, JU0))
        out = []
        for Mq, wq, yq, Bq, iq, vq in zip(Mrows, _weights(batch), _targets(batch), Bw, idx, valid):
            Mw = Mq * wq[:, None]
            out.append((Bq.T @ (Mw[iq] * vq[:, None]), Mw.T @ Mw, Mw.T @ (yq * wq)))
        return ResidualStats(out[0][0], out[1][0], out[2][0],
                             out[0][1], out[1][1], out[2][1],
                             out[0][2], out[1][2], out[2][2])

    def _chunk_stats(self, a, xs):
        theta = from_array(a)
        dsc, Bw, idx, valid, inp = xs
        f = jax.checkpoint(lambda th, b, B, i, v, u: self._batch_stats(th, b, B, i, v, u))

        def body(carry, x):
            return jax.tree.map(jnp.add, carry, f(theta, *x)), None
        L, M = self.prob.cfg.len_basis, self.prob.ind.XM.shape[0]
        zBM, zMM, zM = jnp.zeros((L, M)), jnp.zeros((M, M)), jnp.zeros(M)
        zero = ResidualStats(zBM, zBM, zBM, zMM, zMM, zMM, zM, zM, zM)
        return jax.lax.scan(body, zero, (dsc, Bw, idx, valid, inp))[0]

    def _head_fn(self, a, res):
        return log_marginal_likelihood(from_array(a), assemble_statistics(self.lin, res), self.prob)

    def _chunks(self):
        for s in range(0, self.nb, self.chunk):
            e = min(s + self.chunk, self.nb)
            dsc = jax.tree.map(lambda x: x[s:e], self.ds)
            inp = () if self.pair_only else (jnp.asarray(self.U0[s:e]), jnp.asarray(self.JU0[s:e]))
            yield (dsc, tuple(jnp.asarray(r[s:e]) for r in self.rows),
                   tuple(jnp.asarray(i[s:e]) for i in self.idx),
                   tuple(jnp.asarray(v[s:e]) for v in self.valid), inp)

    def _residual_stats(self, a):
        res = None
        for xs in self._chunks():
            r = self._fwd(a, xs)
            res = r if res is None else jax.tree.map(jnp.add, res, r)
        return res

    # ------------------------------------------------------------------ public
    def __call__(self, a):
        return self._head_value(a, self._residual_stats(a))

    def value_and_grad(self, a):
        a = jnp.asarray(a)
        res = self._residual_stats(a)
        v, (g, ct) = self._head(a, res)
        for xs in self._chunks():
            g = g + self._bwd(a, xs, ct)
        return v, g
