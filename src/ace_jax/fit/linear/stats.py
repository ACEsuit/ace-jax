"""Streamed statistics of weighted design rows in QR form: the generic machinery.

A caller supplies a row function, `rows(batch) -> [(Pw, yw), ...]`, one pair of weighted rows and
weighted targets per quantity (factor), and gets each quantity's triangular factor R and c = Q^T yw:
R^T R is the Gram of its rows and R^T c their right-hand side, never formed.  The merges run on the
host (`host_qr_stream`: LAPACK tpqrt, with checkpoints and MPI shards) or as one lax.scan on a device
(`device_qr_stream`).  Nothing here knows what the rows are; the energy, force and virial rows of a
potential fit are `ace_jax.fit.stats`'.

Rows.  A row generator is a function `rows(batch) -> ((Phi, Y, w), ...)`, one triple per quantity
(a label the caller keeps: "E", "F", "V" for a potential; the blocks of an operator): Phi (n, L) the
design rows, Y (n,) their targets or (n, k) for k right-hand sides that share the rows, w (n,) the
structural weights, 0 on an absent row.  `gram_statistics` and `qr_statistics` stream it over the
batches of a dataset (a pytree whose leaves have a leading batch axis) into one `Gram` or `Factor` per
quantity; `fit.linear.objective` reads those."""
import contextlib
import hashlib
import json
import os
import pathlib
import time

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Gram(NamedTuple):
    """One quantity's statistics in Gram form: G = (w Phi)^T (w Phi) (L, L), b = (w Phi)^T (w Y) (L,) or
    (L, k), yy = |w Y|^2 (per right-hand side), n live rows, logw = sum log w^2 over them."""
    G: jnp.ndarray
    b: jnp.ndarray
    yy: jnp.ndarray
    n: jnp.ndarray
    logw: jnp.ndarray


class Factor(NamedTuple):
    """One quantity's statistics in QR form: R (L, L) upper triangular and c = Q^T (w Y) from a QR of its
    weighted rows w Phi = Q R, so R^T R = G and R^T c = b without forming G (whose condition number is
    kappa(Phi)^2).  yy, n and logw as in `Gram`.  rho: the residual norm |w Y - w Phi x| carried through
    float32 merges (None otherwise), so that an evidence never forms y'y - v'v."""
    R: jnp.ndarray
    c: jnp.ndarray
    yy: jnp.ndarray
    n: jnp.ndarray
    logw: jnp.ndarray
    rho: object = None

    G = property(lambda s: s.R.T @ s.R)
    b = property(lambda s: s.R.T @ s.c)

    def gram(self):
        return Gram(self.G, self.b, self.yy, self.n, self.logw)


def quantities_of(stats):
    """(per-quantity statistics, labels) of a statistics object: its `factors` (or `grams`) and
    `quantities`, or a plain sequence of `Factor` / `Gram` labelled "0", "1", ..."""
    per = getattr(stats, "factors", None)
    if per is None:
        per = getattr(stats, "grams", None)
    if per is None:
        per = tuple(stats)
    return tuple(per), tuple(getattr(stats, "quantities", [str(i) for i in range(len(per))]))


def weighted(Phi, Y, w):
    """(w Phi, w Y, yy, n, logw) of one quantity's rows: the weighted rows and targets a merge takes, and
    the scalars of `Gram` / `Factor`.  Y (n,) or (n, k); yy is then a scalar or (k,)."""
    live = w > 0
    if Y.ndim == 1:
        yw = Y * w
        yy = yw @ yw
    else:
        yw = Y * w[:, None]
        yy = jnp.einsum("nk,nk->k", yw, yw)
    return (Phi * w[:, None], yw, yy, live.sum(),
            jnp.sum(jnp.where(live, 2.0 * jnp.log(jnp.where(live, w, 1.0)), 0.0)))


def gram(Phi, Y, w):
    """The `Gram` of one quantity's rows of one batch."""
    Pw, yw, yy, n, logw = weighted(Phi, Y, w)
    return Gram(Pw.T @ Pw, Pw.T @ yw, yy, n, logw)


def n_batches(ds):
    """Batches of a dataset: its `n_batches`, or the leading axis of its first leaf."""
    n = getattr(ds, "n_batches", None)
    return int(jax.tree.leaves(ds)[0].shape[0]) if n is None else int(n)


def gram_statistics(rows, ds):
    """One `Gram` per quantity of the row generator `rows`, summed over the batches of `ds` (a lax.scan;
    each batch under jax.checkpoint, so a reverse pass recomputes a batch rather than storing them all)."""
    f = jax.checkpoint(lambda b: tuple(gram(*q) for q in rows(b)))
    zero = jax.tree.map(lambda z: jnp.zeros(z.shape), jax.eval_shape(f, jax.tree.map(lambda a: a[0], ds)))
    return jax.lax.scan(lambda c, b: (jax.tree.map(jnp.add, c, f(b)), None), zero, ds)[0]


# Batched merges.  A merge QR-factors the stacked [R c ; Pw yw], (L + r) x (L + 1): ~(4/3) L^3
# whatever r is, plus 2 r L^2 for the new rows (LAPACK geqrf does not exploit R's triangle).
# One merge per batch made that L^3 the whole cost at a large basis: Si o4d20 (L = 5477) has
# 788 packed 224-atom batches of E 117 / F 672 / V 702 rows, 3 merges each, 1.2e15 flops per
# pass with the old explicit Q (5e13 now).  So each quantity's live rows (padding and absent
# virials are zero rows, QR-neutral, and are dropped: E and V are ~2% live there) are buffered
# across batches and merged once at least merge_target (QR_MERGE_ROWS_PER_L x L) have
# accumulated, plus a final flush: per row, (4/3) L^3 / buffer + 2 L^2.  The target column
# rides along as an extra column, so only geqrf's R is formed (its last column is
# Q^T [c ; yw]): no Householder product, which used to double the flops.  That is the device
# form (_qr_merge, device_qr_stream).  The host merges (_host_merge) use LAPACK tpqrt, which keeps
# R triangular: 2 r L^2 alone, no L^3 term; buffering there still amortises tpqrt's panels.
QR_MERGE_ROWS_PER_L = 1


def merge_target(L, merge_rows=None):
    """Live rows that trigger a merge: merge_rows, default QR_MERGE_ROWS_PER_L * L; 0 merges
    every batch (that has live rows)."""
    return int(QR_MERGE_ROWS_PER_L * L if merge_rows is None else merge_rows)


def _live(Pw, yw):
    return jnp.any(Pw != 0, axis=1) | (yw != 0 if yw.ndim == 1 else jnp.any(yw != 0, axis=1))


def _qr_merge(R, c, Pw, yw):
    """Fold weighted rows into a triangular factor: the updating QR of [R ; Pw], with c
    carried as an extra column so that only R is formed (no Q).  mode="raw" is geqrf, which JAX
    cannot differentiate: these statistics are not differentiable (the CPU host path never was)."""
    n = R.shape[1]
    if c.ndim == 2:                                          # k right-hand sides: k extra columns
        A = jnp.concatenate([jnp.concatenate([R, c], 1), jnp.concatenate([Pw, yw], 1)], 0)
        Ra = jnp.triu(jnp.linalg.qr(A, mode="raw")[0].mT[:n])
        return Ra[:, :n], Ra[:, n:]
    A = jnp.concatenate([jnp.concatenate([R, c[:, None]], 1), jnp.concatenate([Pw, yw[:, None]], 1)], 0)
    Ra = jnp.triu(jnp.linalg.qr(A, mode="raw")[0].mT[:n])    # geqrf alone: no Householder product
    return Ra[:, :n], Ra[:, n]


TPQRT_NB = 256     # tpqrt's block size.  mnf148 (32 cores), L rows into L: L 8000 nb 64 / 128 / 256: 195 / 312 / 418
#                    GFLOP/s; L 16000 nb 256 / 512 / 1024: 711 / 632 / 195 (geqrf of the stack: 134 at L 8000)


def _host_merge(R, c, Pw, yw, upper=False):
    """_qr_merge on the host: LAPACK tpqrt, the QR of [[R, c], [Pw, yw]] that keeps R triangular.  c is
    the last column of an (n+1) x (n+1) triangle (zero last row) and only R and c are read back.  geqrf
    of the stacked matrix, which this replaced, treats R as dense: (10/3) L^3 flops per L rows merged
    against 2 L^3, and it ran 5x slower at L = 8000 (12.8 s against 2.5 s, 32 cores).  upper: Pw is
    itself an upper-triangular (n x n) factor (merging two factors, e.g. MPI ranks'): tpqrt then also
    skips its zeros, ~3x faster again."""
    import scipy.linalg.lapack as la
    n, m = R.shape[1], Pw.shape[0]
    if m == 0:
        return np.array(R, np.float64), np.array(c, np.float64)
    k = 1 if np.ndim(c) == 1 else c.shape[1]                 # right-hand sides: k extra columns
    A = np.zeros((n + k, n + k), order="F")
    A[:n, :n], A[:n, n:] = R, np.reshape(c, (n, k))
    B = np.empty((m, n + k), order="F")
    B[:, :n], B[:, n:] = Pw, np.reshape(yw, (m, k))
    a, _, _, info = la.dtpqrt(min(m, n) if upper else 0, min(TPQRT_NB, n + k), A, B, overwrite_a=1, overwrite_b=1)
    if info != 0:
        raise np.linalg.LinAlgError(f"dtpqrt info={info}")
    return np.triu(a[:n, :n]), (a[:n, n].copy() if np.ndim(c) == 1 else a[:n, n:].copy())


def _rank_family(key):
    """The MPI-rank part of a checkpoint key ("-r3of8"), or "" for a serial key: saves clear stale files of their
    own family only, so ranks sharing a directory keep each other's checkpoints."""
    i = key.rfind("-r")
    return key[i:] if i >= 0 and "of" in key[i:] else ""


class QRCheckpoint:
    """Where and how often `host_qr_stream` saves its state (`aj fit --checkpoint DIR`).  The state is
    each factor's (R, c) after flushing its buffered rows, the next batch index and the caller's scalar
    accumulators, in DIR/qrstats-KEY.npz (written to a temporary file, then renamed).  KEY fingerprints
    the problem (`problem_key`), so a checkpoint of a different model, dataset or row dtype is never
    loaded.  A complete pass is kept too: a fit killed after the statistics never repeats them.  Size
    8 L^2 bytes per factor (E, F, V: ~9 GB at L = 2e4); a save costs one extra merge per factor."""

    def __init__(self, directory, every=2700.0, log=print):
        self.dir, self.every, self.log = pathlib.Path(directory), float(every), log

    def path(self, key):
        return self.dir / f"qrstats-{key}.npz"

    def load(self, key, n_factors):
        p = self.path(key)
        if not p.exists():
            return None
        with np.load(p, allow_pickle=False) as z:
            if str(z["key"]) != key or int(z["n_factors"]) != n_factors:
                return None
            acc = [(np.array(z[f"R{k}"]), np.array(z[f"c{k}"])) for k in range(n_factors)]
            return acc, int(z["next"]), json.loads(str(z["extra"]))

    def save(self, key, acc, nxt, extra):
        self.dir.mkdir(parents=True, exist_ok=True)
        fam = _rank_family(key)
        for old in self.dir.glob("qrstats-*.npz"):         # one problem per directory: a stale one is GBs; only
            if old != self.path(key) and _rank_family(old.name[8:-4]) == fam:    # this rank's, not other ranks'
                old.unlink()
        tmp = self.dir / f".qrstats-{key}.tmp.npz"
        np.savez(tmp, key=key, next=nxt, n_factors=len(acc), extra=json.dumps(extra),
                 **{f"{n}{k}": a for k, (R, c) in enumerate(acc) for n, a in (("R", R), ("c", c))})
        os.replace(tmp, self.path(key))


_QR_CHECKPOINT = None      # set by `qr_checkpoint` (the fit pipeline's objective stage); read by linear_qr_statistics


@contextlib.contextmanager
def qr_checkpoint(directory, every=2700.0, log=print):
    """Checkpoint the host QR statistics pass(es) run inside the block to `directory` every `every`
    seconds (QRCheckpoint); None: off.  Scoped, not global: radial learning and ARD, which recompute the
    statistics for every candidate, run outside it."""
    global _QR_CHECKPOINT
    prev = _QR_CHECKPOINT
    _QR_CHECKPOINT = QRCheckpoint(directory, every, log) if directory else None
    try:
        yield _QR_CHECKPOINT
    finally:
        _QR_CHECKPOINT = prev


def problem_key(*trees):
    """A short digest of every array leaf (dtype, shape, bytes) and scalar of `trees`."""
    h = hashlib.blake2b(digest_size=12)
    for leaf in jax.tree.leaves(trees):
        a = np.asarray(leaf)
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes() if a.dtype != object else repr(leaf).encode())
    return h.hexdigest()


def shard_batches(n_batches, rank, size):
    """The batches MPI rank `rank` of `size` streams (`aj fit --mpi`): every size-th, interleaved, so each
    rank sees the dataset's whole range of configuration sizes."""
    return range(rank, n_batches, size)


def merge_factors(a, b):
    """[(R, c), ...] of two disjoint row sets merged into the factors of their union: the stacked QR of
    [R_a; R_b] against [c_a; c_b] (tall-skinny QR).  Exact, as the factors of one pass over both.  Both
    are triangular, so tpqrt skips R_b's zeros too (_host_merge upper=True: ~3x the dense-rows merge)."""
    return [_host_merge(np.asarray(Ra, np.float64), np.asarray(ca, np.float64), np.asarray(Rb, np.float64),
                        np.asarray(cb, np.float64), upper=True) for (Ra, ca), (Rb, cb) in zip(a, b)]


def tree_schedule(size):
    """Rounds of (sender, receiver) pairs merging `size` ranks' factors onto rank 0 in ceil(log2 size)
    rounds: in round k, rank r + 2^k sends to r for every r divisible by 2^(k+1)."""
    rounds, step = [], 1
    while step < size:
        rounds.append([(r + step, r) for r in range(0, size, 2 * step) if r + step < size])
        step *= 2
    return rounds


class MPIWorkerDone(Exception):
    """Raised on MPI ranks other than 0 once their factors are merged: rank 0 runs the rest of the fit."""


_MPI_COMM = None           # set by `mpi_shards` (aj fit --mpi): linear_qr_statistics shards its host pass over it


_MPI_CHUNK = 1 << 27       # elements per Send/Recv: below MPI's 2^31 count for any L


@contextlib.contextmanager
def mpi_shards(comm):
    """Shard the host QR statistics pass(es) run inside the block over the ranks of `comm` (mpi4py); None: off."""
    global _MPI_COMM
    prev = _MPI_COMM
    _MPI_COMM = comm
    try:
        yield comm
    finally:
        _MPI_COMM = prev


def _send_array(comm, a, dest, tag):
    flat = np.ascontiguousarray(a).reshape(-1)
    for k, i in enumerate(range(0, max(flat.size, 1), _MPI_CHUNK)):
        comm.Send(flat[i:i + _MPI_CHUNK], dest=dest, tag=tag + k)


def _recv_array(comm, shape, src, tag):
    flat = np.empty(int(np.prod(shape)), np.float64)
    for k, i in enumerate(range(0, max(flat.size, 1), _MPI_CHUNK)):
        comm.Recv(flat[i:i + _MPI_CHUNK], source=src, tag=tag + k)
    return flat.reshape(shape)


def tree_reduce(acc, extra, comm):
    """Merge every rank's factors `acc` ([(R, c), ...]) and sum its scalar dict `extra` onto rank 0 by
    `tree_schedule`.  Returns (acc, extra) on rank 0 and None elsewhere.  A receiver merges factor by factor:
    peak memory its own factors plus one received factor and its merge workspace (~(1 + 2/3 + 2/3) x its own
    for three equal factors), not two full sets."""
    acc = list(acc)
    rank = comm.Get_rank()
    for rnd in tree_schedule(comm.Get_size()):
        for src, dst in rnd:
            if rank == src:
                comm.send(dict(extra), dest=dst, tag=1)
                for k, (R, c) in enumerate(acc):
                    _send_array(comm, R, dst, 1000 * (2 * k + 1)); _send_array(comm, c, dst, 1000 * (2 * k + 2))
                return None
            if rank == dst:                  # one factor at a time: peak own + one received + its merge workspace
                ex = comm.recv(source=src, tag=1)
                for k, (R, c) in enumerate(acc):
                    other = (_recv_array(comm, R.shape, src, 1000 * (2 * k + 1)),
                             _recv_array(comm, c.shape, src, 1000 * (2 * k + 2)))
                    acc[k] = merge_factors([(R, c)], [other])[0]
                    del other
                extra = {k: extra.get(k, 0.0) + v for k, v in ex.items()}
    return acc, extra


def host_qr_stream(batch_rows_fn, ds, init, merge_rows=None, checkpoint=None, key=None, extra=None, batches=None,
                   progress=None, progress_every=25):
    """The updating QR of streamed rows, its merges on the host (SciPy, the calling thread).
    batch_rows_fn(batch) -> [(Pw, yw), ...] (jitted by the caller), one pair per factor;
    init = [(R0, c0), ...].  Returns [(R, c), ...] as jax arrays.  Each factor's live rows are
    buffered and merged once merge_target(L, merge_rows) have accumulated, then flushed at the
    end ('Batched merges' above); device_qr_stream merges at the same rows.

    Why not a lax.scan of jnp.linalg.qr on the CPU: XLA:CPU calls LAPACK (scipy's OpenBLAS)
    from its own worker threads, and OpenBLAS's multithreaded geqrf, called that way from
    inside a compiled loop, returned NaN on finite input -- deterministically, with 4 or 8
    OpenBLAS threads (a GitHub runner's default: 4 vCPUs) on a ~650-column basis, never with
    1-2 threads nor on the same matrices called directly.  Merges run here never did (tutorial 3,
    #61).  A device (GPU) backend keeps the scan: its QR is not OpenBLAS.

    checkpoint (a QRCheckpoint) with key: resume from a saved state of the same problem and save every
    checkpoint.every seconds (after flushing, so a resumed pass merges exactly as an uninterrupted one
    with the same checkpoints); extra: the caller's dict of scalar accumulators, saved and restored with it.
    batches: the batch indices to stream (default all; an MPI rank's `shard_batches`); the checkpoint
    counts positions in it.  progress(done, total): called every progress_every batches and at the end."""
    order = list(range(n_batches(ds))) if batches is None else list(batches)
    acc = [(np.asarray(R, np.float64), np.asarray(c, np.float64)) for R, c in init]
    buf, fill = [[] for _ in acc], [0] * len(acc)
    start = 0
    if checkpoint is not None:
        st = checkpoint.load(key, len(acc))
        if st is not None:
            acc, start, ex = st
            if extra is not None:
                extra.clear(); extra.update(ex)
            checkpoint.log(f"checkpoint: QR statistics complete ({checkpoint.path(key)})" if start >= len(order)
                           else f"checkpoint: QR statistics resumed at batch {start} of {len(order)}")
    t_save = time.time()
    target = [max(merge_target(R.shape[1], merge_rows), 1) for R, _ in acc]

    def flush(k):
        if fill[k]:
            acc[k] = _host_merge(*acc[k], np.concatenate([p for p, _ in buf[k]]),
                                 np.concatenate([v for _, v in buf[k]]))
        buf[k], fill[k] = [], 0
    for pos in range(start, len(order)):
        i = order[pos]
        parts = batch_rows_fn(jax.tree.map(lambda a: a[i], ds))
        for k, (Pw, yw) in enumerate(parts):
            Pw, yw = np.asarray(Pw, np.float64), np.asarray(yw, np.float64)
            keep = np.any(Pw != 0, axis=1) | (yw != 0 if yw.ndim == 1 else np.any(yw != 0, axis=1))
            if not keep.all():
                Pw, yw = Pw[keep], yw[keep]
            buf[k].append((Pw, yw)); fill[k] += len(yw)
            if fill[k] >= target[k]:
                flush(k)
        if progress is not None and ((pos + 1) % progress_every == 0 or pos + 1 == len(order)):
            progress(pos + 1, len(order))
        if checkpoint is not None and pos + 1 < len(order) and time.time() - t_save >= checkpoint.every:
            tf = time.time()
            for k in range(len(acc)):
                flush(k)
            ts = time.time()
            checkpoint.save(key, acc, pos + 1, extra or {})
            checkpoint.log(f"checkpoint: saved batch {pos + 1} of {len(order)}: flush {ts - tf:.0f} s, "
                           f"write {time.time() - ts:.0f} s")
            t_save = time.time()
    for k in range(len(acc)):
        flush(k)
    if checkpoint is not None and start < len(order):
        checkpoint.save(key, acc, len(order), extra or {})
    return [(jnp.asarray(R), jnp.asarray(c)) for R, c in acc]


CPU_SCAN_MAX_L = 512     # above this, device_qr_stream on the CPU backend warns (#61, host_qr_stream)


def device_qr_stream(rows_fn, ds, init, extra0, merge_rows=None):
    """host_qr_stream as one lax.scan over the batches, for a device backend.  rows_fn(batch)
    -> ([(Pw, yw), ...], extra); extra (a pytree like extra0) is summed.  Shapes are static, so
    each factor carries a (target + r, L) row buffer and a fill count: a batch's r rows are
    compacted live-first (a stable sort) and written at the fill, which then advances by the
    live count only; a lax.cond merges the buffer (rows past the fill masked to zero) once the
    fill reaches the target, so at most target + r rows are ever pending.  The same merge
    points as host_qr_stream.  Returns ([(R, c), ...], extra).

    Not for the CPU backend beyond small L: XLA:CPU's geqrf inside the scan can return NaN with
    multithreaded OpenBLAS (#61; host_qr_stream says why), so callers route the CPU to the host
    and this warns when L > CPU_SCAN_MAX_L there."""
    if jax.default_backend() == "cpu" and max(jnp.shape(R)[1] for R, _ in init) > CPU_SCAN_MAX_L:
        import warnings
        warnings.warn("device_qr_stream on the CPU backend: XLA:CPU's geqrf in a scan can return NaN "
                      "with multithreaded OpenBLAS at this basis size (#61); use host_qr_stream", stacklevel=2)
    shapes = jax.eval_shape(rows_fn, jax.tree.map(lambda a: a[0], ds))[0]
    init = [(jnp.asarray(R), jnp.asarray(c)) for R, c in init]
    plan = [(max(merge_target(R.shape[1], merge_rows), 1), P.shape[0]) for (R, _), (P, _) in zip(init, shapes)]
    bufs0 = [(jnp.zeros((t + r, R.shape[1]), R.dtype), jnp.zeros((t + r,) + c.shape[1:], R.dtype), jnp.zeros((), jnp.int32))
             for (t, r), (R, c) in zip(plan, init)]

    def merge(R, c, BP, By, n):
        m = jnp.arange(BP.shape[0]) < n                         # rows past the fill: stale, masked
        return _qr_merge(R, c, jnp.where(m[:, None], BP, 0.0), jnp.where(m if By.ndim == 1 else m[:, None], By, 0.0))

    def body(carry, batch):
        fac, bufs, extra = carry
        parts, ex = rows_fn(batch)
        fac2, bufs2 = [], []
        for (t, _), (R, c), (BP, By, n), (Pw, yw) in zip(plan, fac, bufs, parts):
            live = _live(Pw, yw)
            o = jnp.argsort(~live, stable=True)                  # live rows first, in order
            BP = jax.lax.dynamic_update_slice(BP, Pw[o].astype(BP.dtype), (n, jnp.zeros((), jnp.int32)))
            By = jax.lax.dynamic_update_slice(By, yw[o].astype(By.dtype), (n,) if By.ndim == 1 else (n, jnp.zeros((), jnp.int32)))
            n = n + live.sum(dtype=jnp.int32)
            R, c, n = jax.lax.cond(n >= t, lambda a: (*merge(*a), jnp.zeros((), jnp.int32)),
                                   lambda a: (a[0], a[1], a[4]), (R, c, BP, By, n))
            fac2.append((R, c)); bufs2.append((BP, By, n))
        return (fac2, bufs2, jax.tree.map(jnp.add, extra, ex)), None

    (fac, bufs, extra), _ = jax.lax.scan(body, (init, bufs0, extra0), ds)
    out = [jax.lax.cond(n > 0, lambda a: merge(*a), lambda a: (a[0], a[1]), (R, c, BP, By, n))
           for (R, c), (BP, By, n) in zip(fac, bufs)]                # the final flush
    return out, extra


def active_checkpoint():
    """The QRCheckpoint of the enclosing `qr_checkpoint` block, or None."""
    return _QR_CHECKPOINT


def active_comm():
    """The MPI communicator of the enclosing `mpi_shards` block, or None."""
    return _MPI_COMM


# ---------------------------------------------------------------------------
# Factors of a row generator.  `wrows(batch) -> [(Pw, yw, yy, n, logw), ...]` is the generator's rows
# through `weighted`, one tuple per quantity; `weighted_rows(rows)` makes it.

def weighted_rows(rows):
    """The weighted form of a row generator: batch -> [weighted(Phi, Y, w) for each quantity]."""
    return lambda batch: [weighted(*q) for q in rows(batch)]


def _scalar_names(label, yy):
    """Names of one quantity's scalar accumulators (checkpointed, summed over MPI ranks): yy, n, logw;
    one yy per right-hand side when there are several."""
    k = int(np.prod(np.shape(yy)))
    return [f"yy_{label}"] if np.ndim(yy) == 0 else [f"yy_{label}.{j}" for j in range(k)]


def host_qr_statistics(wrows, ds, L, labels, merge_rows=None, key=None):
    """One `Factor` per quantity from weighted rows, merged on the host (`host_qr_stream`): the pass that
    `qr_checkpoint` checkpoints and `mpi_shards` spreads over MPI ranks (rank 0 returns; the others raise
    MPIWorkerDone once their factors are merged).  wrows: batch -> [(Pw, yw, yy, n, logw), ...], jitted by
    the caller; labels: the quantities' names (they name the checkpointed scalars and the progress lines).
    key: a function returning the problem's fingerprint (`problem_key`), called only under a checkpoint."""
    shapes = jax.eval_shape(wrows, jax.tree.map(lambda a: a[0], ds))
    zeros = [(np.zeros((L, L)), np.zeros((L,) + tuple(o[1].shape[1:]))) for o in shapes]
    yy_names = [_scalar_names(q, np.zeros(o[2].shape)) for q, o in zip(labels, shapes)]
    scal = {k: 0.0 for q, names in zip(labels, yy_names) for k in (*names, f"n_{q}", f"logw_{q}")}

    def parts(batch):
        out = wrows(batch)
        for q, names, (_, _, yy, n, lw) in zip(labels, yy_names, out):
            for name, v in zip(names, np.reshape(np.asarray(yy, np.float64), -1)):
                scal[name] += float(v)
            scal[f"n_{q}"] += float(n); scal[f"logw_{q}"] += float(lw)
        return [(Pw, yw) for Pw, yw, *_ in out]
    ck, comm = _QR_CHECKPOINT, _MPI_COMM
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    ckey = base = None if ck is None else key()
    if ckey is not None and size > 1:
        ckey = f"{ckey}-r{rank}of{size}"         # each rank checkpoints its own shard
    t0 = time.time()

    def report(done, total):                 # per-rank throughput: stderr on every rank, progress.jsonl on 0
        import sys
        dt = time.time() - t0
        print(f"qr stats rank {rank}/{size}: {done}/{total} batches, {dt:.0f} s "
              f"({3600.0 * done / max(dt, 1e-9):.1f} batches/h)", file=sys.stderr, flush=True)
        from .progress import emit
        emit("qr_batches", rank=rank, size=size, done=done, total=total, seconds=round(dt, 1))
    nb = n_batches(ds)
    acc = host_qr_stream(parts, ds, zeros, merge_rows, checkpoint=ck, key=ckey, extra=scal,
                         batches=shard_batches(nb, rank, size), progress=report)
    if size > 1:
        out = tree_reduce(acc, scal, comm)
        if out is None:
            raise MPIWorkerDone(rank)
        acc, merged = out
        scal.update(merged)
        if ck is not None:          # the merged factors under the serial key: a later run on one process (the
            t = time.time()         # MAP job of the two-job form, `--stats-only`) finds the statistics complete
            ck.save(base, acc, nb, scal)
            ck.log(f"checkpoint: merged statistics of {size} ranks saved ({ck.path(base)}, {time.time() - t:.0f} s)")

    def yy(q, names, o):
        v = [scal[k] for k in names]
        return jnp.asarray(v[0]) if o[2].shape == () else jnp.asarray(v).reshape(o[2].shape)
    return tuple(Factor(jnp.asarray(R), jnp.asarray(c), yy(q, names, o), jnp.asarray(scal[f"n_{q}"]),
                        jnp.asarray(scal[f"logw_{q}"]))
                 for q, names, o, (R, c) in zip(labels, yy_names, shapes, acc))


def device_qr_statistics(wrows, ds, L, merge_rows=None):
    """`host_qr_statistics` as one traceable lax.scan (`device_qr_stream`), for a device backend; the caller
    jits it (with what the rows read as arguments, so one compile serves every call of the same shapes)."""
    shapes = jax.eval_shape(wrows, jax.tree.map(lambda a: a[0], ds))
    zeros = [(np.zeros((L, L)), np.zeros((L,) + tuple(o[1].shape[1:]))) for o in shapes]
    z0 = jnp.zeros(())

    def rows_fn(batch):
        out = wrows(batch)
        return [(Pw, yw) for Pw, yw, *_ in out], [o[2:] for o in out]
    fac, ex = device_qr_stream(rows_fn, ds, zeros, [(jnp.zeros(o[2].shape), z0, z0) for o in shapes], merge_rows)
    return tuple(Factor(R, c, *e) for (R, c), e in zip(fac, ex))


def qr_statistics(rows, ds, L, labels=None, merge_rows=None, host=None):
    """One `Factor` per quantity of the row generator `rows` over the batches of `ds`: on the host on the
    CPU backend (`host_qr_stream` says why), as a compiled scan on a device.  labels: the quantities' names
    (default "0", "1", ...).  merge_rows: `merge_target`."""
    wrows = weighted_rows(rows)
    if host is None:
        host = jax.default_backend() == "cpu"
    if not host:
        return jax.jit(lambda d: device_qr_statistics(wrows, d, L, merge_rows))(ds)
    nq = len(jax.eval_shape(wrows, jax.tree.map(lambda a: a[0], ds)))
    labels = [str(i) for i in range(nq)] if labels is None else list(labels)
    return host_qr_statistics(jax.jit(wrows), ds, L, labels, merge_rows,
                              key=lambda: problem_key(ds, L, merge_target(L, merge_rows), labels))
