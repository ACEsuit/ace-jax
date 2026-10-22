"""Recursive (DAG) product basis: each order-k AA monomial as one order-(k-1) node
times one A entry, so ~1 whole-row multiply per monomial instead of k - 1.

The AA monomials ace-jax contracts are real (A is real: real Y_lm, and for ACE the
coupling A2B / ctilde, for PACE the T map and ctilde, act on AA *after* the
product), so the recursion applies directly to the uncoupled monomials
prod_t A[s_t], as in ML-PACE (`ACECTildeBasisSet`'s rank-wise products of
`ACERecursiveEvaluator`: each basis function's parent is an order-(k-1) node,
auxiliary nodes inserted where no basis function serves) and symmetrix-xl's
`multivariate_polynomial` (the same "parent x one factor" DAG, levels by degree).

A plan (`build_plan`, host-side numpy on static index arrays) is a tuple of levels.
Level 1 is A itself (feature-major rows, no gather).  Level k >= 2 is the node set
N_k with, per node, a parent in N_{k-1} and an A row: L_k = L_{k-1}[parent] * A[a].
Nodes are the needed order-k monomials plus the auxiliary ones some order-(k+1)
monomial needs as a parent.  Each level's nodes are sorted by (parent, a), so the
gather of L_{k-1} reads its rows in order (its adjoint scatter-adds in order).  The
readout map sends each AA column (concatenated `specs` order) to its node; duplicate
monomials (equal multisets in one order) share a node and their coefficients add.

Parent choice (`mode`):
  "prefix"  the parent of (s_0..s_{k-1}) is its prefix (s_0..s_{k-2}).  The
            multiply order is that of `product_basis_t`'s chain, so every
            monomial is bitwise the current one.
  "greedy"  monomials as multisets; a parent is any (k-1)-sub-multiset already
            in N_{k-1} (a basis monomial or an auxiliary node), and where none
            exists a greedy set cover picks the auxiliary node serving the most
            unresolved monomials.  Fewer nodes; values to roundoff.

Evaluation: `dag_levels` (the node rows per level) and `dag_contract`, the
readout sum_k ct_k^T L_k (or per node with per-node columns), with ct_k the
AA coefficients scattered onto level k's nodes (zero on auxiliary nodes).
"""
import heapq
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

PLAN_MODES = ("prefix", "greedy")
# aa_dag="auto" takes the DAG ("greedy") from this many chain-form multiplies per site (sum over orders of
# n_monomials * (order - 1)) and keeps the chain form below.  Timed in LAMMPS (lammps-jax), one model at a time
# in both forms on the same node (bench/accuracy_cost/envs/sulis-{cpu,gpu}-dagpair.sbatch, Si accuracy-cost
# ladders): below ~2,200 the two tie within noise (0.98-1.06x, CPU node and A100); from ~6,200 the DAG wins
# on both (CPU 1.24-2.55x, A100 1.12-1.84x at 7,095-175,021).  The chain form below keeps small models bitwise.
AUTO_MIN_CHAIN_MUL = 4000


def _parents_prefix(keys_k):
    return {key: (key[:-1], key[-1]) for key in keys_k}


def _parents_greedy(keys_k, have):
    """{node: (parent, a)} for multiset keys (sorted tuples) of one order; `have` is the
    set of order-(k-1) keys that exist (mutated: auxiliary parents are added to it)."""
    out, todo = {}, []
    for key in keys_k:
        cands = [key[:i] + key[i + 1:] for i in reversed(range(len(key)))
                 if i == len(key) - 1 or key[i] != key[i + 1]]
        hit = next((c for c in cands if c in have), None)
        if hit is not None:
            i = next(i for i in reversed(range(len(key))) if key[:i] + key[i + 1:] == hit)
            out[key] = (hit, key[i])
        else:
            todo.append((key, cands))
    if not todo:
        return out
    by_cand = defaultdict(list)                       # set cover: candidate -> unresolved nodes
    for j, (_, cands) in enumerate(todo):
        for c in cands:
            by_cand[c].append(j)
    count = {c: len(v) for c, v in by_cand.items()}
    heap = [(-n, c) for c, n in count.items()]
    heapq.heapify(heap)
    done = np.zeros(len(todo), bool)
    while heap:
        n, c = heapq.heappop(heap)
        if -n != count[c]:                            # stale entry: re-push with its live count
            if count[c]:
                heapq.heappush(heap, (-count[c], c))
            continue
        if not count[c]:
            continue
        have.add(c)
        for j in by_cand[c]:
            if done[j]:
                continue
            done[j] = True
            key = todo[j][0]
            i = next(i for i in reversed(range(len(key))) if key[:i] + key[i + 1:] == c)
            out[key] = (c, key[i])
            for c2 in todo[j][1]:
                count[c2] -= 1
        count[c] = 0
    return out


def build_plan(specs, n_A, mode="greedy"):
    """The DAG plan for the product basis of `specs` (one (n_v, k) index array per
    correlation order, the `product_basis_t` argument) over n_A A rows.

    Returns ((parent, a) per level k >= 2, (col, node) readout per level k >= 1, n_AA):
    int32 numpy arrays.  The readout of level k maps the AA columns of order k to
    level-k node ids (level 1: the A row)."""
    if mode not in PLAN_MODES:
        raise ValueError(f"mode must be one of {PLAN_MODES}, got {mode!r}")
    specs = [np.asarray(s, np.int64) for s in specs]
    K = max((s.shape[1] for s in specs if s.size), default=1)
    canon = (lambda r: tuple(r)) if mode == "prefix" else (lambda r: tuple(sorted(r)))
    cols = defaultdict(list)                          # order -> [(AA column, key)]
    off = 0
    for s in specs:
        for j, row in enumerate(s.tolist()):
            cols[s.shape[1]].append((off + j, canon(row)))
        off += s.shape[0]
    n_AA = off
    keys = {k: {key for _, key in cols[k]} for k in range(1, K + 1)}
    keys[1] |= {(i,) for i in range(n_A)}             # level 1 is all of A
    parent = {}
    for k in range(K, 1, -1):                         # top down: parents may add aux nodes below
        par = (_parents_prefix(keys[k]) if mode == "prefix"
               else _parents_greedy(keys[k], keys[k - 1]))
        if mode == "prefix":
            keys[k - 1] |= {p for p, _ in par.values()}
        parent[k] = par
    ids = {1: {(i,): i for i in range(n_A)}}
    levels = []
    for k in range(2, K + 1):                         # bottom up: ids sorted by (parent id, a)
        rows = sorted((ids[k - 1][p], a, key) for key, (p, a) in parent[k].items())
        ids[k] = {key: i for i, (_, _, key) in enumerate(rows)}
        levels.append((np.asarray([r[0] for r in rows], np.int32),
                       np.asarray([r[1] for r in rows], np.int32)))
    readout = []
    for k in range(1, K + 1):
        c = np.asarray([cc for cc, _ in cols[k]], np.int32)
        nd = np.asarray([ids[k][key] for _, key in cols[k]], np.int32)
        readout.append((c, nd))
    return tuple(levels), tuple(readout), n_AA


def plan_stats(specs, plan):
    """Counts: monomials, DAG nodes (aux = nodes not read out), and the per-node
    gathers / multiplies of the forward pass for the chain form vs the DAG."""
    levels, readout, n_AA = plan
    chain_mul = sum(int(s.shape[0]) * (int(s.shape[1]) - 1) for s in specs)
    chain_gather = sum(int(s.size) for s in specs)
    nodes = sum(int(p.shape[0]) for p, _ in levels)
    read_nodes = sum(len(set(nd.tolist())) for _, nd in readout[1:])
    return {"n_AA": n_AA, "dag_nodes": nodes, "aux_nodes": nodes - read_nodes,
            "chain_mul": chain_mul, "chain_gather": chain_gather,
            "dag_mul": nodes, "dag_gather": 2 * nodes,
            "per_level": [int(p.shape[0]) for p, _ in levels]}


def to_device(plan):
    """The plan's index arrays as int32 jax arrays (model leaves)."""
    levels, readout, n_AA = plan
    i32 = lambda a: jnp.asarray(a, jnp.int32)
    return (tuple((i32(p), i32(a)) for p, a in levels),
            tuple((i32(c), i32(nd)) for c, nd in readout))


def dag_levels(At, dag):
    """[L_1 = At, L_2, ..., L_K]: node rows (n_k, n) of feature-major A (n_A, n).
    Plain gathers: `indices_are_sorted` / `promise_in_bounds` hints were neutral on
    the CPU and made the GPU's adjoint 37x slower (o4d20 product stage, RTX 4000
    Ada: 908 vs 24.5 ms fwd+bwd)."""
    L = [At]
    for p, a in dag[0]:
        L.append(L[-1][p] * At[a])
    return L


# CPU: columns (nodes) per chunk of `dag_contract`.  The levels are a few times n_AA rows
# of n columns, so unchunked they stream through memory (o4d20: 68k rows, 0.5 GB at
# n = 1024, as slow as a contiguous multiply of that size: bandwidth-bound).  A chunk is
# checkpointed, so its backward recomputes it while it is in cache: 1.38x on the
# o4d20 product stage at 16 (8: 1.24x, 64: 1.22x; Xeon Silver 4216, one core).
CPU_CHUNK = 16


def _contract_cols(At, dag, cks, node_z):
    L = dag_levels(At, dag)
    hi = jax.lax.Precision.HIGHEST
    out = 0.0
    for k, ((_c, nd), ck) in enumerate(zip(dag[1], cks)):
        if ck is None:
            continue
        Lk = At[nd] if k == 0 else L[k]               # order 1: the few A rows read, not all of A
        if node_z is None:
            out = out + jnp.matmul(ck.T, Lk, precision=hi)
        else:
            out = out + jnp.einsum("ai,ai->i", Lk, ck[:, node_z])
    return out


def _contract_chunked(At, dag, cks, node_z, c):
    n_A, n = At.shape
    nb = -(-n // c)
    if nb <= 1:
        return _contract_cols(At, dag, cks, node_z)
    pad = nb * c - n
    Ab = jnp.pad(At, ((0, 0), (0, pad))).reshape(n_A, nb, c).transpose(1, 0, 2)
    body = jax.checkpoint(lambda a, z: _contract_cols(a, dag, cks, z))
    if node_z is None:
        out = jax.lax.map(lambda a: body(a, None), Ab)                  # (nb, Q, c)
        return out.transpose(1, 0, 2).reshape(out.shape[1], -1)[:, :n]
    zb = jnp.pad(node_z, (0, pad)).reshape(nb, c)
    return jax.lax.map(lambda x: body(*x), (Ab, zb)).reshape(-1)[:n]


def dag_contract(At, dag, ct, node_z=None, chunk="auto"):
    """The product-basis readout without forming AA: with node_z None,
    (Q, n) = sum_k ct_k^T L_k, ct (n_AA, Q) (PACE's rho); with node_z, (n,) =
    sum_k sum_a ct_k[a, node_z[i]] L_k[a, i] (ACE's folded site energy, the
    `product_basis_dot` form).  ct_k is ct scattered onto level k's nodes (zero on
    auxiliary nodes, duplicates added); order 1 reads its A rows directly.

    chunk: "auto" evaluates in checkpointed column chunks of CPU_CHUNK on the CPU
    (`lax.platform_dependent`) and unchunked elsewhere; an int forces that chunk,
    None no chunking."""
    cks = []
    for k, (c, nd) in enumerate(dag[1]):
        if c.shape[0] == 0:
            cks.append(None)
        elif k == 0:
            cks.append(ct[c])
        else:
            cks.append(jnp.zeros((dag[0][k - 1][0].shape[0], ct.shape[1]), ct.dtype).at[nd].add(ct[c]))
    if chunk is None:
        return _contract_cols(At, dag, cks, node_z)
    if chunk != "auto":
        return _contract_chunked(At, dag, cks, node_z, int(chunk))
    return jax.lax.platform_dependent(
        At, cpu=lambda At: _contract_chunked(At, dag, cks, node_z, CPU_CHUNK),
        default=lambda At: _contract_cols(At, dag, cks, node_z))


_PLANS = {}                                           # (mode, specs content) -> device plan


def _plan_for(specs, mode):
    """`build_plan` on the device, memoised on the specs' content: a readout-only
    model swap (`ACECalculator.model`) does not rebuild it (0.5 s at o4d20)."""
    arrs = [np.asarray(s) for s in specs]
    key = (mode,) + tuple((a.shape, a.tobytes()) for a in arrs)
    if key not in _PLANS:
        if len(_PLANS) >= 16:
            _PLANS.pop(next(iter(_PLANS)))
        n_A = 1 + max((int(a.max()) for a in arrs if a.size), default=0)
        _PLANS[key] = to_device(build_plan(arrs, n_A, mode))
    return _PLANS[key]


def chain_multiplies(specs):
    """Per-site multiplies of the chain-form product basis: sum over orders of n_monomials * (order - 1)."""
    return sum(int(np.shape(a)[0]) * max(int(np.shape(a)[1]) - 1, 0) for a in specs if np.size(a))


def resolve_aa_dag(model, mode):
    """The plan mode `mode` stands for on `model`: "auto" is "greedy" when the chain form's multiplies
    (`chain_multiplies` of its aa_specs) reach AUTO_MIN_CHAIN_MUL, else None (the chain form); a model
    without a product basis to plan (no `aa_dag` field) resolves to None; other modes as given ("auto" never
    resolves to "cueq", which is opt-in)."""
    if mode != "auto":
        return mode
    specs = getattr(model, "aa_specs", None) if hasattr(model, "aa_dag") else None
    return "greedy" if specs is not None and chain_multiplies(specs) >= AUTO_MIN_CHAIN_MUL else None


def with_aa_dag(model, mode="greedy"):
    """`model` with DAG plans attached for its energy path's product basis (mode None: plans removed, cueq
    plans included; "auto": `resolve_aa_dag`; "cueq": the folded readout's product basis through
    cuequivariance instead, `eval.cueq.with_cueq`, a single-species or species-embedding lean ACEModel only).  ACEModel: `aa_dag`
    for `aa_specs` (the sparse layout and the unblocked dense one) and, on a lean model, `blk_aa_dag` for
    `blk_aa_specs`; read by the folded readout only (the basis methods keep the chain form), so apply it after
    `lean`.  PACEModel: `aa_dag`.  A model without the fields (a wrapper reading the basis) is returned as
    given.  Host-side, on concrete index arrays."""
    import dataclasses
    if not hasattr(model, "aa_dag"):
        return model
    mode = resolve_aa_dag(model, mode)
    has_blk = hasattr(model, "blk_aa_dag")
    if mode is None:
        kw = {"blk_aa_dag": None} if has_blk else {}
        if hasattr(model, "cueq_plan"):
            kw.update(cueq_plan=None, blk_cueq_plan=None)
        return dataclasses.replace(model, aa_dag=None, **kw)
    if mode == "cueq":                            # cuequivariance (`eval.cueq`), opt-in, never "auto"
        from .cueq import with_cueq
        return with_cueq(model)
    if mode not in PLAN_MODES:
        raise ValueError(f"aa_dag must be one of {PLAN_MODES}, 'cueq', 'auto' or None, got {mode!r}")
    kw = {"aa_dag": _plan_for(model.aa_specs, mode)}
    if has_blk:
        kw["blk_aa_dag"] = None if model.blk_aa_specs is None else _plan_for(model.blk_aa_specs, mode)
    return dataclasses.replace(model, **kw)
