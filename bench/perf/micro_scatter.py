"""Micro-benchmark: ways to sum the product-basis adjoint rows back into A.

The adjoint of AA = prod_t A[s[:, t]] w.r.t. A is dA[a] = sum over positions
p with s_p = a of Q_p, Q (P, n) the per-position contributions.  Compared, on
the index pattern of a real model (P positions -> n_A targets, n nodes):

  scatter_nm   node-major  (n, P) -> (n, n_A) scatter-add along axis 1
  scatter_fm   feature-major (P, n) -> (n_A, n) scatter-add of whole rows
  segsum_sorted  rows permuted so targets are sorted, segment_sum(indices_are_sorted)
  tree         rows permuted by target, then a static pairwise tree reduction:
               each level is a gather of two row sets and an add (no atomics)
  onehot_mm    dA = S^T @ Q with the (n_A, P) incidence matrix

    PYTHONPATH=bench:src python bench/perf/micro_scatter.py Cantor_medium 8192
"""
import json
import sys
import time

import numpy as np


def tree_levels(tgt):
    """Static index arrays for a pairwise segmented reduction of rows sorted
    by `tgt`.  Level L maps the current rows to ceil(c/2) rows per segment:
    new[r] = cur[a[r]] + cur[b[r]], b = -1 -> the appended zero row.  Returns
    (perm, levels, final order of targets)."""
    perm = np.argsort(tgt, kind="stable")
    seg = tgt[perm]
    levels = []
    while True:
        # segment of each current row; pair consecutive rows within a segment
        a_idx, b_idx, new_seg = [], [], []
        i = 0
        n = len(seg)
        while i < n:
            j = i
            while j < n and seg[j] == seg[i]:
                j += 1
            for k in range(i, j, 2):
                a_idx.append(k)
                b_idx.append(k + 1 if k + 1 < j else -1)
                new_seg.append(seg[i])
            i = j
        if len(a_idx) == len(seg):                     # every segment is one row
            break
        levels.append((np.asarray(a_idx, np.int32), np.asarray(b_idx, np.int32)))
        seg = np.asarray(new_seg)
    return perm.astype(np.int32), levels, seg


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ace_jax.eval.pace_build import build_basis
    from ace_jax.eval.pace_io import parse_yace

    model, n = sys.argv[1], int(sys.argv[2])
    _, a = parse_yace(f"bench/scaling/models/pace_{model}.yace")
    b = build_basis(a)
    nA = len(a["Z"]) * b["n_a_local"]
    tgt = np.concatenate([np.asarray(s).T.ravel() for s in b["aa_specs"]]).astype(np.int32)
    P = len(tgt)
    perm, levels, final_seg = tree_levels(tgt)
    assert np.array_equal(final_seg, np.unique(tgt))
    out = {"model": model, "n": n, "P": P, "n_A": nA, "tree_levels": len(levels)}
    rng = np.random.default_rng(0)
    Qfm = jnp.asarray(rng.standard_normal((P, n)))
    Qnm = Qfm.T.copy()
    t = jnp.asarray(tgt)
    tp = jnp.asarray(tgt[perm])
    S = jnp.zeros((nA, P)).at[t, jnp.arange(P)].set(1.0)
    lv = [(jnp.asarray(x), jnp.asarray(y)) for x, y in levels]
    uniq = jnp.asarray(final_seg)

    def tree(Q):
        cur = Q[jnp.asarray(perm)]
        for x, y in lv:
            curz = jnp.concatenate([cur, jnp.zeros((1, cur.shape[1]), cur.dtype)])
            cur = curz[x] + curz[y]
        return jnp.zeros((nA, Q.shape[1]), Q.dtype).at[uniq].set(cur)

    fns = {
        "scatter_nm": (jax.jit(lambda Q: jnp.zeros((Q.shape[0], nA), Q.dtype).at[:, t].add(Q)), Qnm),
        "scatter_fm": (jax.jit(lambda Q: jnp.zeros((nA, Q.shape[1]), Q.dtype).at[t].add(Q)), Qfm),
        "segsum_sorted": (jax.jit(lambda Q: jax.ops.segment_sum(
            Q[jnp.asarray(perm)], tp, num_segments=nA, indices_are_sorted=True)), Qfm),
        "tree": (jax.jit(tree), Qfm),
        "onehot_mm": (jax.jit(lambda Q: jnp.matmul(S, Q, precision=jax.lax.Precision.HIGHEST)), Qfm),
    }
    ref = np.asarray(fns["scatter_fm"][0](Qfm))
    for k, (f, Q) in fns.items():
        try:
            r = f(Q)
            jax.block_until_ready(r)
        except Exception as ex:                                    # noqa: BLE001
            out[k] = {"error": repr(ex)[:160]}
            continue
        rr = np.asarray(r).T if k == "scatter_nm" else np.asarray(r)
        err = float(np.max(np.abs(rr - ref)))
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            jax.block_until_ready(f(Q))
            ts.append(time.perf_counter() - t0)
        out[k] = {"ms": 1e3 * float(np.median(ts)), "max_err": err}
    out["Q_MB"] = P * n * 8 / 1e6
    print(json.dumps(out))


if __name__ == "__main__":
    main()
