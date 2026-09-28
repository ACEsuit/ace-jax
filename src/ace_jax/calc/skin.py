"""Skin (Verlet) neighbour list for repeated calls on moving atoms.

The (n, K_skin) neighbour matrix is built for cutoff + skin and reused until an
atom has moved more than skin / 2 since the build, or the cell, pbc, species or
atom count change (LAMMPS: `neighbor 1.0 bin`, `neigh_modify every 1 delay 0
check yes`).  Each call is then one host->device copy (displacements) and one
compiled `step`: it forms rij = rij0 + u[j] - u[i] over the skin list, compacts
it to the K slots inside the cutoff (what ML-PACE's ComputeNeigh does), and
evaluates E, F and V.  The step also returns the displacement and overflow
flags, so the caller can rebuild and call again when either is set.

The step takes the displacements u = pos - pos0 since the build (differenced on
the host in float64), not the positions: rij0 is the list's own edge vector,
rounded to the model dtype once, and u is small, so in float32 rij is as
accurate as a fresh list's.  x[j] - x[i] + shift in float32 would carry the
rounding of |x| and |shift| (a box length), not of |rij|.

Forces are assembled in skin-slot space: the reverse edge of skin slot (i, s),
j = idx_s[i, s], sits at skin slot rev_s[i, s] of row j (`nlist.reverse_slots`,
matched per periodic image), and at compact slot csum[j, rev] - 1 of the model's
(n, K) graph, so F is a gather.  When reverse_slots cannot match every edge (a
rounding-boundary case) rev_s is None and the step scatters instead.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from ..eval.nlist import dense_from_sparse, dense_graph, have_matscipy_neighbours
from ..eval.nlist import backend as nlist_backend
from ..eval.nlist import reverse_slots, sparse_graph


def round_k(k, step=4):
    """Dense row capacity: the largest neighbour count rounded up, so a small
    change in the count keeps the compiled shape (and the neighbour_matrix K)."""
    return max(step, -(-int(k) // step) * step)


def species_lut(elements):
    """119-entry atomic number -> model species index table; -1 where absent."""
    lut = np.full(119, -1, np.int32)
    for i, z in enumerate(elements):
        lut[int(z)] = i
    return lut


@dataclasses.dataclass(frozen=True)
class SkinState:
    """A skin list and what it was built for.  Host copies (pos0, cell, pbc,
    numbers) serve `valid` and the displacements; the device arrays are the
    step's inputs."""
    pos0: np.ndarray          # (n, 3) positions at the build, float64
    cell: np.ndarray
    pbc: np.ndarray
    numbers: np.ndarray
    idx_s: jax.Array          # (n, K_skin) neighbour index, 0 on padded slots
    rij0_s: jax.Array         # (n, K_skin, 3) edge vectors at the build (periodic
                              # image included), parked at rc + skin on padded slots
    live_s: jax.Array         # (n, K_skin) slot holds a skin neighbour
    rev_s: jax.Array | None   # (n, K_skin) slot of the reverse edge in row idx_s; None: scatter
    node_z: jax.Array         # (n,) species index
    half_skin: jax.Array      # () skin / 2, model dtype
    K_skin: int
    K: int                    # compact (in-cutoff) row capacity: the step's static shape
    n_edges: int              # in-cutoff edges at the build
    k_max: int                # largest in-cutoff row count at the build
    backend: str              # what built the list

    @property
    def arrays(self):
        return (self.idx_s, self.rij0_s, self.live_s, self.rev_s, self.node_z, self.half_skin)

    def displacements(self, pos, dtype):
        """The step's input: pos - pos0, in float64 on the host, then `dtype`."""
        return np.ascontiguousarray(np.asarray(pos, float) - self.pos0, dtype)


def build(pos, cell, pbc, numbers, lut, rc, skin, K_skin_hint=None, device=None, *,
          dtype=np.float64, K_hint=None):
    """A SkinState for these positions.  K_skin_hint / K_hint are capacities
    learnt by an earlier build (kept when still large enough, so the compiled
    shape does not change); None or too small relearns from a sparse list.
    Raises ValueError naming any element the model does not have."""
    pos = np.array(pos, float)
    numbers = np.asarray(numbers)
    z = np.where(numbers < len(lut), lut[np.minimum(numbers, len(lut) - 1)], -1)
    if (z < 0).any():
        bad = sorted({int(v) for v in numbers[z < 0]})
        raise ValueError(f"elements {bad} not in the model")
    rcs = float(rc) + float(skin)
    n = len(pos)
    dg, backend, Ks = None, None, K_skin_hint
    if Ks:
        try:
            dg = dense_graph(pos, cell, pbc, rcs, Ks, device=device)
            backend = "neighbour_matrix" if have_matscipy_neighbours() else nlist_backend()
        except ValueError:                             # an atom outgrew K_skin
            dg = None
    if dg is None:
        g = sparse_graph(pos, cell, pbc, rcs)
        Ks = round_k(np.bincount(np.asarray(g.senders), minlength=n).max(initial=0))
        dg = dense_from_sparse(g, rcs, max_neighbours=Ks)
        backend = nlist_backend()
    idx = np.asarray(dg.idx).astype(np.int64)
    rij = np.asarray(dg.rij, float)
    count = np.asarray(dg.count)
    live = np.arange(Ks)[None, :] < count[:, None]
    idx = np.where(live, idx, 0)
    r2 = np.sum(rij * rij, axis=-1)
    n_cut = np.sum(live & (r2 < rc * rc), axis=1)
    k_max = int(n_cut.max(initial=0))
    K = max(int(K_hint or 0), round_k(k_max))
    try:
        rev = jnp.asarray(reverse_slots(idx, rij, count), jnp.int32)
    except ValueError:                                 # a rounding-boundary mismatch
        rev = None
    return SkinState(
        pos0=pos, cell=np.array(cell, float), pbc=np.array(pbc, bool), numbers=numbers.copy(),
        idx_s=jnp.asarray(idx, jnp.int32), rij0_s=jnp.asarray(rij, dtype),
        live_s=jnp.asarray(live), rev_s=rev,
        node_z=jnp.asarray(z, jnp.int32), half_skin=jnp.asarray(0.5 * skin, dtype),
        K_skin=int(Ks), K=K, n_edges=int(n_cut.sum()), k_max=k_max, backend=backend)


def valid(state, pos, cell, pbc, numbers, skin):
    """Whether `state` still holds every pair within the cutoff: same atom count,
    cell, pbc and species, and no atom moved more than skin / 2 since the build
    (two atoms then approach by at most skin)."""
    if state is None or np.shape(pos) != state.pos0.shape:
        return False
    if not (np.array_equal(np.asarray(cell, float), state.cell)
            and np.array_equal(np.asarray(pbc, bool), state.pbc)
            and np.array_equal(numbers, state.numbers)):
        return False
    d2 = np.max(np.sum((np.asarray(pos, float) - state.pos0) ** 2, axis=1), initial=0.0)
    return bool(d2 <= (0.5 * skin) ** 2)


# packed output: ([E, F (3n), V (9)] in the model dtype, [drift, overflow, k_max,
# n_edges] in int32 -- float32 holds integers exactly only to 2**24)
def unpack(out, n):
    values, counts = jax.device_get(out)
    E, F, V = float(values[0]), values[1:1 + 3 * n].reshape(n, 3), values[1 + 3 * n:].reshape(3, 3)
    drift, overflow, k_max, n_edges = (int(c) for c in counts)
    return E, F, V, bool(drift), bool(overflow), k_max, n_edges


def step(model, u, arrays, rc, K):
    """Skin list -> compact (n, K) graph -> packed ((E, F, V), (drift, overflow,
    k_max, n_edges)); see `unpack`.  u (n, 3): displacements since the build
    (`SkinState.displacements`).  Traceable; `jitted_step` compiles it.  drift:
    an atom has moved more than skin / 2; overflow: a row has more than K
    neighbours inside the cutoff.  Either means the result is not to be used."""
    idx_s, rij0_s, live_s, rev_s, node_z, half_skin = arrays
    Ks = idx_s.shape[1]
    rij_s = rij0_s + (u[idx_s] - u[:, None, :])
    r2 = jnp.sum(rij_s * rij_s, axis=-1)
    valid_s = live_s & (r2 < rc * rc)
    # in-cutoff slots first, in their skin order: compact slot k takes the
    # (k+1)-th valid entry, found by a search in the row's running count (gathers
    # only; a per-row argsort was ~1 ms at 8k atoms on an A100)
    csum = jnp.cumsum(valid_s.astype(jnp.int32), axis=1)
    cnt = csum[:, -1]
    want = jnp.arange(1, K + 1, dtype=jnp.int32)
    order = jax.vmap(lambda c: jnp.searchsorted(c, want, side="left"))(csum)
    order = jnp.minimum(order, Ks - 1)
    # slots past the row's count are padding, whatever `order` clipped to
    mask = want[None, :] <= cnt[:, None]
    idx = jnp.where(mask, jnp.take_along_axis(idx_s, order, axis=1), 0)
    pad = jnp.asarray([1.0, 0.0, 0.0], u.dtype) * model.pad_cutoff()
    rij = jnp.where(mask[..., None], jnp.take_along_axis(rij_s, order[..., None], axis=1), pad)
    zi = jnp.broadcast_to(node_z[:, None], idx.shape)
    E, g_r, V = model.energy_forces_virial_dense(rij, zi, node_z[idx], idx, mask, node_z,
                                                 return_edge_grad=True)
    if rev_s is None:
        F = g_r.sum(axis=1).at[idx.reshape(-1)].add(-g_r.reshape(-1, 3))
    else:
        # F_i = sum_k g[i, k] - sum over edges j -> i of g[j, that edge's compact
        # slot].  Every edge j -> i is the reverse of a live skin slot (i, s):
        # it sits at skin slot rev_s[i, s] of row j, which is in the model's
        # graph iff valid there, at compact slot csum - 1.  Testing the reverse
        # slot's own validity (not (i, s)'s) keeps F consistent with the energy
        # even when rounding puts the two directions on either side of rc.
        # The two row sums nearly cancel (exactly, on a symmetric site), so they
        # accumulate in float64 where JAX has it: in float32 their rounding was
        # ~2x a fresh list's force error on zero-force cells (test_perf_parity).
        comp = jnp.where(valid_s, csum - 1, 0)
        back_valid = live_s & valid_s[idx_s, rev_s]
        back = jnp.where(back_valid[..., None], g_r[idx_s, comp[idx_s, rev_s]], 0.0)
        acc = jnp.float64 if jax.config.jax_enable_x64 else g_r.dtype
        F = (g_r.astype(acc).sum(axis=1) - back.astype(acc).sum(axis=1)).astype(g_r.dtype)
    drift = jnp.max(jnp.sum(u * u, axis=1), initial=0.0) > half_skin ** 2
    overflow = jnp.any(cnt > K)
    counts = jnp.stack([drift.astype(jnp.int32), overflow.astype(jnp.int32),
                        jnp.max(cnt, initial=0), jnp.sum(cnt)]).astype(jnp.int32)
    return jnp.concatenate([E[None], F.ravel(), V.ravel()]), counts


def jitted_step(static, rc):
    """`step` compiled for a model structure and cutoff `rc`: f(params, u, arrays,
    K=K) -> packed, where (params, static) = eqx.partition(model, eqx.is_array).
    One jax.jit over the model's arrays (C++ dispatch), passed per call so new
    weights of the same structure take effect without a retrace; the static part
    and rc are closed over, and matmul precision is fixed at trace time."""
    import equinox as eqx
    rc = float(rc)

    def packed(params, u, arrays, K):
        with jax.default_matmul_precision("highest"):
            return step(eqx.combine(params, static), u, arrays, rc, K)

    return jax.jit(packed, static_argnames=("K",))
