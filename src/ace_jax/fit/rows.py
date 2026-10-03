"""Feature rows Phi = [linear block | residual block] for one padded batch.

Linear block: energy row = sum of site descriptors; force/virial rows from the
edge Jacobian, species-blocked exactly as ACEpotentials.site_descriptors so the
rows equal ACEfit's design matrix (tests/test_gp_rows.py).

Residual block: energy row = sum_i k(x_i, x_m); force/virial rows from
dk/dx . dX/dr + dk/ds . ds/dr, scanned over node chunks so the (chunk, M, D)
kernel-gradient tensor -- 8 GB for a whole batch at target scale -- never
exists at once."""
import math
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .data import VOIGT, flat_edges
from .kernels import grad_k_rows, k_rows
from .summary import summary_edge_jacobian


class Rows(NamedTuple):
    E: jnp.ndarray   # (C, Dt)
    F: jnp.ndarray   # (Ncap, 3, Dt)
    V: jnp.ndarray   # (C, 6, Dt)


def _place(dst, src, z, cfg):
    """Add compact (..., D) rows into species block z of a (..., len_basis) array,
    in the layout of ACEModel.site_descriptors (B blocks first, then pair blocks)."""
    nB, nP, NZ = cfg.n_B, cfg.n_pair, cfg.NZ
    dst = dst.at[..., z * nB:(z + 1) * nB].add(src[..., :nB])
    return dst.at[..., NZ * nB + z * nP: NZ * nB + (z + 1) * nP].add(src[..., nB:])


def _voigt(T, rij):
    """T (E, ..., 3) derivative w.r.t. rij (E, 3) -> the 6 Voigt virial
    contributions (E, ..., 6): V_ab = -1/2 (T_a r_b + T_b r_a)."""
    cols = [-0.5 * (T[..., a] * rij[:, b].reshape((-1,) + (1,) * (T.ndim - 2))
                    + T[..., b] * rij[:, a].reshape((-1,) + (1,) * (T.ndim - 2)))
            for a, b in VOIGT]
    return jnp.stack(cols, axis=-1)


def _e0_columns(E, node_z, node_mask, node_cfg, cfg):
    """With cfg.e0_cols, the last NZ energy columns are the configs' atom counts per species
    (an E0 shift is a constant site energy: it moves no force or virial row)."""
    if not getattr(cfg, "e0_cols", False):
        return E
    C, L0 = E.shape[0], cfg.len_readout
    onehot = jax.nn.one_hot(node_z, cfg.NZ) * node_mask[:, None]
    counts = jax.ops.segment_sum(onehot, node_cfg, num_segments=C + 1)[:C]
    return E.at[:, L0:].set(counts)


def linear_rows(model, cfg, batch):
    Ncap, K = batch.nbr.shape
    C = batch.y_E.shape[0]          # configs per batch is a property of the batch, not of cfg
    rij, send, recv, mask = flat_edges(batch.rij, batch.nbr, batch.nbr_mask)
    zi = batch.node_z[send]
    # dense per-node contraction: the sparse edge_jacobian gathers dB/dA onto
    # every edge, 14 GB at the Cantor basis size (n_B 1348, 18k edges/batch)
    X, J = model.edge_jacobian_dense(batch.rij, jnp.broadcast_to(batch.node_z[:, None], (Ncap, K)),
                                     batch.node_z[batch.nbr], batch.nbr_mask)      # (Ncap, D), (E, D, 3)
    seg = lambda a, ids, n: jax.ops.segment_sum(a, ids, num_segments=n)
    L = cfg.len_basis
    E = jnp.zeros((Ncap, L))
    F = jnp.zeros((Ncap, 3, L))
    V = jnp.zeros((C + 1, 6, L))
    edge_cfg = batch.node_cfg[send]
    for z in range(cfg.NZ):
        E = _place(E, jnp.where((batch.node_z == z)[:, None], X, 0.0), z, cfg)
        Jz = jnp.where((zi == z)[:, None, None], J, 0.0)
        dEdr = seg(Jz, recv, Ncap) - seg(Jz, send, Ncap)             # (Ncap, D, 3)
        F = _place(F, -jnp.swapaxes(dEdr, 1, 2), z, cfg)
        Vz = seg(_voigt(Jz, rij), edge_cfg, C + 1)                   # (C+1, D, 6)
        V = _place(V, jnp.swapaxes(Vz, 1, 2), z, cfg)
    E = _e0_columns(seg(E, batch.node_cfg, C + 1)[:C], batch.node_z, batch.node_mask, batch.node_cfg, cfg)
    return Rows(E, F, V[:C]), X, J


ROWS_EDGE_BUDGET = int(os.environ.get("ACEJAX_ROWS_EDGE_BUDGET", 1 << 28))
"""Elements of the largest per-chunk temporary the training-side rows may build at once (2 GiB in
f64): the edge Jacobian J (nc*K, D, 3) or, if wider, the edge-feature intermediates of
`edge_jacobian_dense` (`_node_elems`).  A batch that fits takes the unchunked `linear_rows` path
(compiled shapes unchanged); a larger one -- size-aware packing gives n_cap ~ 3.3k nodes with
k_cap ~ 900 slots, a 220 GB J -- is node-chunked (`rows_node_chunk`).  Read at trace time, so a
test can lower it (or ACEJAX_ROWS_EDGE_BUDGET, to run a whole suite chunked) to force chunking."""

_NODE_ELEMS = {}


def _max_elems(jaxpr):
    """Largest element count of any value of a jaxpr, sub-jaxprs included."""
    from jax.extend import core as jcore
    best = 0
    for eqn in jaxpr.eqns:
        for v in eqn.outvars:
            shape = getattr(v.aval, "shape", None)
            if shape is not None:
                best = max(best, math.prod(shape))
        for p in eqn.params.values():
            for sub in (p if isinstance(p, (tuple, list)) else (p,)):
                if isinstance(sub, jcore.ClosedJaxpr):
                    best = max(best, _max_elems(sub.jaxpr))
                elif isinstance(sub, jcore.Jaxpr):
                    best = max(best, _max_elems(sub))
    return best


def _model_key(model):
    """A hashable structural key of a model: tree structure plus each leaf's shape/dtype (array)
    or value (static) -- equal for the fresh models radial learning builds every call (whose leaves
    may be tracers), so the per-node trace is shared and no model is kept alive."""
    leaves, tdef = jax.tree.flatten(model)
    key = [tdef]
    for x in leaves:
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            key.append((tuple(x.shape), str(x.dtype)))
        else:
            key.append(x)
    key = tuple(key)
    hash(key)
    return key


def _node_elems(model, cfg, K):
    """Peak temporary elements per centre node of `edge_jacobian_dense` with K neighbour slots:
    the larger of J's K*D*3 and the widest intermediate of one node's abstract trace (edge
    features can be wider than J: 10x on the si fixture, 3x at the production Cantor basis).
    Every temporary scales linearly with the node count, so nc nodes cost nc times this.
    Traced on an abstract copy of the model (float leaves as ShapeDtypeStructs), once per
    structural model key and K."""
    import equinox as eqx
    try:
        key = (_model_key(model), K)
    except TypeError:               # an unhashable static leaf: no caching
        key = None
    if key is not None and key in _NODE_ELEMS:
        return _NODE_ELEMS[key]
    n = K * cfg.D * 3
    dyn, static = eqx.partition(model, eqx.is_inexact_array)
    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), dyn)
    r = jax.ShapeDtypeStruct((1, K, 3), jnp.zeros(()).dtype)
    i = jax.ShapeDtypeStruct((1, K), jnp.int32)
    m = jax.ShapeDtypeStruct((1, K), bool)
    try:
        jp = jax.make_jaxpr(lambda d, *a: eqx.combine(d, static).edge_jacobian_dense(*a))(abstract, r, i, i, m)
        n = max(n, _max_elems(jp.jaxpr))
    except NotImplementedError:     # a model without a B-basis (PACE): never reaches the rows
        pass
    if key is not None:
        _NODE_ELEMS[key] = n
    return n


ROW_GRANULE = 8
"""Node-chunk granule of the bounded rows."""


def rows_node_chunk(model, cfg, Ncap, K):
    """The node chunk the bounded rows use for an (Ncap, K) batch: None (unchunked) when the
    whole batch's peak temporary (Ncap * `_node_elems`) fits ROWS_EDGE_BUDGET, else the largest
    multiple of ROW_GRANULE that fits -- at least one granule.  At the production Cantor basis
    with k_cap = 928 one node's temporaries are ~26M elements, so the budget gives 8 nodes per
    chunk (one ~1.6 GB temporary); the residual block sub-scans gcd(nc, cfg.node_chunk) nodes."""
    per = _node_elems(model, cfg, K)
    if Ncap * per <= ROWS_EDGE_BUDGET:
        return None
    return max(ROW_GRANULE, (ROWS_EDGE_BUDGET // per) // ROW_GRANULE * ROW_GRANULE)


def linear_rows_bounded(model, cfg, batch, node_chunk=None):
    """`linear_rows(model, cfg, batch)[0]` with peak memory bounded independent of n_cap: the
    unchunked path when the batch fits ROWS_EDGE_BUDGET, `linear_rows_chunked` otherwise.
    node_chunk forces a chunk.  Every caller that only needs the rows (not X, J) uses this."""
    nc = (rows_node_chunk(model, cfg, *batch.nbr.shape) if node_chunk is None
          else node_chunk)
    return linear_rows(model, cfg, batch)[0] if nc is None else linear_rows_chunked(model, cfg, batch, nc)


def chunked_rows_fn(model, cfg, node_chunk=None):
    """`batch -> linear_rows_bounded(model, cfg, batch, node_chunk)`, jitted: compiled once per
    batch shape and reused.  Called eagerly, linear_rows_chunked's fori_loop is retraced on every
    call with that batch's arrays baked into the body as constants -- an XLA compile per batch.
    Build this once per (model, cfg) and loop batches through it.  node_chunk None: the
    ROWS_EDGE_BUDGET policy (`rows_node_chunk`); an int always chunks."""
    return jax.jit(lambda b: linear_rows_bounded(model, cfg, b, node_chunk=node_chunk))


def linear_rows_chunked(model, cfg, batch, node_chunk=256):
    """`linear_rows(model, cfg, batch)[0]` computed a chunk of centre nodes at a time.

    `linear_rows` materialises the edge Jacobian J (Ncap*K, D, 3) of the whole batch.  For one
    cell of more than ~2.8k atoms at the production Cantor basis (D = 3007, K ~ 85) that tensor
    and the GEMM building it exceed 2^31 elements, and XLA's int32-indexed GEMM autotuning fails.
    Here each chunk's J (node_chunk*K, D, 3) is built, scattered into E/F/V and dropped; the rows
    are identical (tests/test_rows_chunked.py).  Returns Rows only (no X, J)."""
    return _rows_scan(model, cfg, batch, node_chunk)[0]


def _rows_scan(model, cfg, batch, node_chunk, with_X=False, proj=None, resid=None):
    """One fori_loop over chunks of `node_chunk` centre nodes building, per chunk, the edge
    Jacobian J (nc*K, D, 3) and everything derived from it, so nothing of size Ncap*K*D exists.

    Returns (lin, X, JU0, res):
      lin  Rows: `linear_rows(...)[0]`;
      X    (Ncap, D) compact descriptors if with_X (else None);
      JU0  (Ncap, K, d, 3) = proj^T J if proj (D, d) is given (else None) -- the projected
           Jacobian the residual inputs / derivative DTC / host cache need, d wide not D;
      res  Rows: `residual_rows(theta, spec, ind, cfg, batch, X, J)` if resid = (theta, spec, ind)
           with M > 0 (else None), its own scan over sub-chunks of gcd(node_chunk,
           cfg.node_chunk) nodes run inside each chunk.
    Identical to the unchunked functions up to summation order (tests/test_rows_bounded.py)."""
    Ncap, K = batch.nbr.shape
    C = batch.y_E.shape[0]
    L, D = cfg.len_basis, cfg.D
    nc = int(min(node_chunk, Ncap))
    n_chunks = -(-Ncap // nc)
    pad = n_chunks * nc - Ncap
    Np = Ncap + pad
    rij = jnp.concatenate([batch.rij, jnp.broadcast_to(batch.rij[-1:], (pad, K, 3))])
    nbr = jnp.concatenate([batch.nbr, jnp.zeros((pad, K), batch.nbr.dtype)])
    msk = jnp.concatenate([batch.nbr_mask, jnp.zeros((pad, K), bool)])
    node_z = jnp.concatenate([batch.node_z, jnp.zeros(pad, batch.node_z.dtype)])
    node_cfg = jnp.concatenate([batch.node_cfg, jnp.full(pad, C, batch.node_cfg.dtype)])
    seg = lambda a, ids, n: jax.ops.segment_sum(a, ids, num_segments=n)
    local = jnp.repeat(jnp.arange(nc), K)
    sl = lambda a, s0: jax.lax.dynamic_slice_in_dim(a, s0, nc)

    if resid is not None and resid[2].XM.shape[0] == 0:
        resid = None                    # BLR limit: no residual block
    if resid is not None:
        from .feature import dwarp, warp_u
        theta, spec, ind = resid
        M = ind.XM.shape[0]
        c = math.gcd(nc, int(cfg.node_chunk))       # residual sub-chunk: divides nc, <= node_chunk
        rf, sf, _, mf = flat_edges(rij, nbr, msk)
        s_all, Js_all = summary_edge_jacobian(rf, jnp.repeat(jnp.arange(Np), K), Np,
                                              cfg.r0, cfg.rcut, cfg.p, mf)
        Js_all = Js_all.reshape(Np, K, 3)
    Pres = None if resid is None else resid[2].Pmap

    def contrib(i):
        """Chunk i's contributions, every one entering the accumulators linearly.  Checkpointed:
        under reverse mode the loop then saves only i per chunk and rebuilds the chunk's J (and
        residual sub-scan) in the backward pass, instead of stacking every chunk's J-sized
        residuals -- O(Ncap*K*D) again (tests/test_rows_bounded.py, gradient variant)."""
        s0 = i * nc
        r_c, nb_c, m_c, z_c = sl(rij, s0), sl(nbr, s0), sl(msk, s0), sl(node_z, s0)
        X, J = model.edge_jacobian_dense(r_c, jnp.broadcast_to(z_c[:, None], (nc, K)), node_z[nb_c], m_c)
        recv = nb_c.reshape(-1)
        zi, r_flat, edge_cfg = z_c[local], r_c.reshape(-1, 3), node_cfg[s0 + local]
        E_c = jnp.zeros((nc, L))
        F = jnp.zeros((Np, 3, L))         # receive side: any node
        F_c = jnp.zeros((nc, 3, L))       # send side: the senders are this chunk's own centre nodes
        V = jnp.zeros((C + 1, 6, L))
        for z in range(cfg.NZ):
            E_c = _place(E_c, jnp.where((z_c == z)[:, None], X, 0.0), z, cfg)
            Jz = jnp.where((zi == z)[:, None, None], J, 0.0)
            F = _place(F, -jnp.swapaxes(seg(Jz, recv, Np), 1, 2), z, cfg)
            F_c = _place(F_c, jnp.swapaxes(seg(Jz, local, nc), 1, 2), z, cfg)
            V = _place(V, jnp.swapaxes(seg(_voigt(Jz, r_flat), edge_cfg, C + 1), 1, 2), z, cfg)
        out = {"lin": (E_c, F, F_c, V)}
        J4 = J.reshape(nc, K, D, 3)
        if with_X:
            out["X"] = X
        if proj is not None:
            out["JU0"] = jnp.einsum("nkDa,Dq->nkqa", J4, proj)
        if resid is not None:
            U0 = X @ Pres
            JU = dwarp(U0, ind.warp)[:, None, :, None] * jnp.einsum("nkDa,Dq->nkqa", J4, Pres)
            U = warp_u(U0, ind.warp)
            sc, Jsc = sl(s_all, s0), sl(Js_all, s0)
            Kx = k_rows(theta, spec, U, sc, z_c, ind.XM, ind.SM, ind.ZM, ind.embed)    # (nc, M)
            sub = lambda a: a.reshape((nc // c, c) + a.shape[1:])
            chunks = (sub(U), sub(sc), sub(z_c), sub(JU), sub(Jsc), sub(r_c), sub(nb_c), sub(m_c),
                      sub(s0 + jnp.arange(nc)), sub(sl(node_cfg, s0)))
            (FR, VR), _ = jax.lax.scan(lambda cr, ch: (_residual_step(theta, spec, ind, cr, ch), None),
                                       (jnp.zeros((Np, 3, M)), jnp.zeros((C + 1, 6, M))), chunks)
            out["res"] = (Kx, FR, VR)
        return out

    contrib = jax.checkpoint(contrib)
    upd = lambda a, x, s0: jax.lax.dynamic_update_slice_in_dim(a, x, s0, 0)

    def body(i, acc):
        s0 = i * nc
        o = contrib(i)
        Enodes, F, V = acc["lin"]
        E_c, Fr, F_c, Vc = o["lin"]
        F = F + Fr
        F = upd(F, sl(F, s0) + F_c, s0)
        out = dict(acc, lin=(upd(Enodes, E_c, s0), F, V + Vc))
        if with_X:
            out["X"] = upd(acc["X"], o["X"], s0)
        if proj is not None:
            out["JU0"] = upd(acc["JU0"], o["JU0"], s0)
        if resid is not None:
            ER, FR, VR = acc["res"]
            Kx, FRc, VRc = o["res"]
            out["res"] = (upd(ER, Kx, s0), FR + FRc, VR + VRc)
        return out

    init = {"lin": (jnp.zeros((Np, L)), jnp.zeros((Np, 3, L)), jnp.zeros((C + 1, 6, L)))}
    if with_X:
        init["X"] = jnp.zeros((Np, D))
    if proj is not None:
        init["JU0"] = jnp.zeros((Np, K, proj.shape[1], 3))
    if resid is not None:
        init["res"] = (jnp.zeros((Np, M)), jnp.zeros((Np, 3, M)), jnp.zeros((C + 1, 6, M)))
    acc = jax.lax.fori_loop(0, n_chunks, body, init)
    Enodes, F, V = acc["lin"]
    E = _e0_columns(seg(Enodes, node_cfg, C + 1)[:C], batch.node_z, batch.node_mask, batch.node_cfg, cfg)
    lin = Rows(E, F[:Ncap], V[:C])
    res = None
    if resid is not None:
        ER, FR, VR = acc["res"]
        res = Rows(seg(ER, node_cfg, C + 1)[:C], FR[:Ncap], VR[:C])
    return (lin, acc["X"][:Ncap] if with_X else None,
            acc["JU0"][:Ncap] if proj is not None else None, res)


def _residual_step(theta, spec, ind, carry, ch):
    """One cfg.node_chunk sub-chunk of the residual force/virial rows: accumulates the chunk's
    -dE/dr into F (n, 3, M) and its virial into V (C+1, 6, M).  Shared by
    residual_rows_from_inputs and the chunked `_rows_scan`."""
    F, V = carry
    Uc, sc, zc, JUc, Jsc, rc, nc, mc, nodes, cfgc = ch
    c, K = mc.shape
    M = ind.XM.shape[0]
    dKx, dKs = grad_k_rows(theta, spec, Uc, sc, zc, ind.XM, ind.SM, ind.ZM, ind.embed)  # (c,M,d), (c,M)
    T = (jnp.einsum("ckqa,cmq->ckam", JUc, dKx)
         + jnp.einsum("cka,cm->ckam", Jsc, dKs))                          # (c, K, 3, M)
    T = jnp.where(mc[:, :, None, None], T, 0.0)
    F = F.at[nodes].add(T.sum(1))                                          # -dE/dr_i = +sum_k T
    F = F.at[nc.reshape(-1)].add(-T.reshape(c * K, 3, M))                  # -dE/dr_j = -T
    Vv = _voigt(jnp.swapaxes(T, 2, 3).reshape(c * K, M, 3), rc.reshape(c * K, 3))  # (cK, M, 6)
    V = V.at[jnp.repeat(cfgc, K)].add(jnp.swapaxes(Vv, 1, 2))
    return F, V


def residual_inputs(ind, cfg, batch, X):
    """The residual GP's site inputs for one batch: feature-mapped coordinates
    U = phi_res(X) (Ncap, d), summaries s (Ncap,) and the summary edge Jacobian
    Js (E, 3).  Shared by residual_rows and predict's DTC term."""
    from .feature import apply
    Ncap = batch.nbr.shape[0]
    rij, send, _, mask = flat_edges(batch.rij, batch.nbr, batch.nbr_mask)
    s, Js = summary_edge_jacobian(rij, send, Ncap, cfg.r0, cfg.rcut, cfg.p, mask)
    return apply(X, ind.Pmap, ind.warp), s, Js


def residual_rows(theta, spec, ind, cfg, batch, X, J):
    """Residual rows from the full compact descriptors X (Ncap, D) and their
    edge Jacobian J (E, D, 3): projects onto the feature map and defers to
    residual_rows_from_inputs."""
    Ncap, K = batch.nbr.shape
    if ind.XM.shape[0] == 0:        # BLR limit: no residual block (static shape, legal under jit)
        C = batch.y_E.shape[0]
        return Rows(jnp.zeros((C, 0)), jnp.zeros((Ncap, 3, 0)), jnp.zeros((C, 6, 0)))
    U0 = X @ ind.Pmap
    JU0 = jnp.einsum("nkDa,Dq->nkqa", J.reshape(Ncap, K, cfg.D, 3), ind.Pmap)   # (Ncap, K, d, 3)
    return residual_rows_from_inputs(theta, spec, ind, cfg, batch, U0, JU0)


def pair_feature_inputs(model, ind, cfg, batch):
    """(U0, JU0) for a pair-density feature map (Pmap zero outside the n_pair pair
    rows) from the pair radials alone -- no many-body basis.  Equals
    (X @ Pmap, Pmap^T J) of the full path (tests/test_gp_hostcache.py)."""
    Ncap, K = batch.nbr.shape
    zi = jnp.broadcast_to(batch.node_z[:, None], (Ncap, K))
    Xp, dRp = model.pair_features_dense(batch.rij, zi, batch.node_z[batch.nbr], batch.nbr_mask)
    P = ind.Pmap[cfg.n_B:]                                            # (n_pair, d)
    return Xp @ P, jnp.einsum("nkpa,pq->nkqa", dRp.reshape(Ncap, K, -1, 3), P)


def residual_rows_from_inputs(theta, spec, ind, cfg, batch, U0, JU0):
    """Residual rows from the projected feature coordinates U0 = X @ Pmap
    (Ncap, d) and their edge Jacobian JU0 = Pmap^T J (Ncap, K, d, 3), pre-warp."""
    from .feature import dwarp, warp_u
    Ncap, K = batch.nbr.shape
    C, c = batch.y_E.shape[0], cfg.node_chunk
    M, d = ind.XM.shape                 # d = feature dim (D for isotropic, n_pair for density)
    if M == 0:
        return Rows(jnp.zeros((C, 0)), jnp.zeros((Ncap, 3, 0)), jnp.zeros((C, 6, 0)))
    rij, send, _, mask = flat_edges(batch.rij, batch.nbr, batch.nbr_mask)
    s, Js = summary_edge_jacobian(rij, send, Ncap, cfg.r0, cfg.rcut, cfg.p, mask)
    U = warp_u(U0, ind.warp)                                          # (Ncap, d)
    # feature edge Jacobian JU (Ncap, K, d, 3) = dwarp(U0) * (Pmap^T J);
    # folds ind.scale, the projection and the warp into one object.
    JU = dwarp(U0, ind.warp)[:, None, :, None] * JU0
    Kx = k_rows(theta, spec, U, s, batch.node_z, ind.XM, ind.SM, ind.ZM, ind.embed)   # (Ncap, M)
    E = jax.ops.segment_sum(Kx, batch.node_cfg, num_segments=C + 1)[:C]

    step = lambda carry, ch: (_residual_step(theta, spec, ind, carry, ch), None)

    n_chunks = Ncap // c
    chunk = lambda a: a.reshape((n_chunks, c) + a.shape[1:])
    chunks = (chunk(U), chunk(s), chunk(batch.node_z), chunk(JU),
              chunk(Js.reshape(Ncap, K, 3)), chunk(batch.rij), chunk(batch.nbr),
              chunk(batch.nbr_mask), chunk(jnp.arange(Ncap)), chunk(batch.node_cfg))
    (F, V), _ = jax.lax.scan(step, (jnp.zeros((Ncap, 3, M)), jnp.zeros((C + 1, 6, M))), chunks)
    return Rows(E, F, V[:C])


def _cat_rows(lin, res):
    return Rows(jnp.concatenate([lin.E, res.E], 1), jnp.concatenate([lin.F, res.F], 2),
                jnp.concatenate([lin.V, res.V], 2))


def _no_res(batch):
    C, Ncap = batch.y_E.shape[0], batch.nbr.shape[0]
    return Rows(jnp.zeros((C, 0)), jnp.zeros((Ncap, 3, 0)), jnp.zeros((C, 6, 0)))


def batch_rows_parts(theta, spec, model, ind, cfg, batch, with_X=False, with_JU0=False):
    """(lin, res, X, JU0) of one batch with peak memory bounded by ROWS_EDGE_BUDGET:
    `linear_rows` + `residual_rows` when the batch fits, the fused node-chunked `_rows_scan`
    otherwise.  X (Ncap, D) if with_X, JU0 = Pmap^T J (Ncap, K, d, 3) if with_JU0 (else None)."""
    nc = rows_node_chunk(model, cfg, *batch.nbr.shape)
    if nc is None:
        lin, X, J = linear_rows(model, cfg, batch)
        res = residual_rows(theta, spec, ind, cfg, batch, X, J)
        Ncap, K = batch.nbr.shape
        JU0 = (jnp.einsum("nkDa,Dq->nkqa", J.reshape(Ncap, K, cfg.D, 3), ind.Pmap)
               if with_JU0 else None)
        return lin, res, (X if with_X else None), JU0
    lin, X, JU0, res = _rows_scan(model, cfg, batch, nc, with_X=with_X,
                                  proj=ind.Pmap if with_JU0 else None, resid=(theta, spec, ind))
    return lin, (_no_res(batch) if res is None else res), X, JU0


def batch_rows(theta, spec, model, ind, cfg, batch):
    lin, res, _, _ = batch_rows_parts(theta, spec, model, ind, cfg, batch)
    return _cat_rows(lin, res)
