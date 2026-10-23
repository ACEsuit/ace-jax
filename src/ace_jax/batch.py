"""Batched evaluation: energies, forces and stresses of many structures in one compiled call.

Small cells leave a GPU (or a CPU's cores, with the C++ kernel) idle when evaluated one per call: the
dispatch and the neighbour list cost more than the model.  Here structures are padded into per-size
buckets (`pack`), each batch's neighbour lists are built on the device in one call (`neighbours`) and the
whole batch is evaluated as one dense block of rows (`efv`); `BatchCalculator` does all three for a list
of ASE Atoms.  Design: docs/dev/specs/2026-10-23-batched-evaluation-design.md.

The layout is padded (B, N): positions (B, N, 3), the live atoms first in each structure.  That is the
shape JAX-MD's integrators take under vmap, so the same pieces serve batched MD and relaxation driven by
JAX-MD (ace-jax hosts no integrators).

Neighbour lists: all pairs over the periodic images within the cutoff (`n_k = ceil(rc / w_k)` per
periodic axis, w_k the perpendicular width), so any cell works, smaller than the cutoff included, and
any periodicity per structure.  That is O(N^2 S) per structure -- right for small cells, wrong for
large ones, which `BatchCalculator` evaluates one at a time through `ACECalculator` (`N_DEVICE`) -- and
slow on a CPU, where lists are built per structure on the host instead (`host_neighbours`).  When
matscipy-neighbours has the batched XLA FFI build (libAtoms/matscipy-neighbours#11) it is used on
either device (`builder="auto"`).
"""

import dataclasses
import math
import time
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .eval.splinify import AUTO

# Structures above this many atoms go through ACECalculator one at a time: the all-pairs build is
# O(N^2 S) per structure, and a large cell already fills the device on its own.
N_DEVICE = 256
# The same with the JAX model on a CPU, where batching gains only on the smallest cells (lestrade, o4d12
# float64: 4x at 2 atoms, 1.5x at 8, slower than one cell per call from 64; XLA's CPU backend does not
# spread the rows over cores, the C++ kernel does).
N_DEVICE_CPU_JAX = 16
# builder="auto" on a CPU without the batched FFI build: all pairs on the device for at most this many
# candidates per atom (N S, padded), a cell list per structure on the host above (lestrade, C++ kernel:
# all pairs 5 / 3.4 us per atom against 39 / 13 for 2- / 8-atom cells; 15 against 5.6 at 64 atoms).
CPU_ALLPAIRS_MAX = 512
# Default cap on the padded atoms of one batch (B x N).  B for a bucket is the largest power of two
# within it (and no larger than the bucket's structure count rounded up to one).
MAX_BATCH_ATOMS = 2 ** 14
# Bytes the JAX builder may hold for one lax.map step: its (b, N, N*S, 3) image distances and copies.
SCRATCH_BYTES = 256 * 2 ** 20


def bucket_size(n):
    """The padded size for n atoms (or images): a power of two up to 8 (a 2-atom cell is not padded to
    8), a multiple of 8 up to 128 (< 11% padding above 64), then quarter-octave steps rounded up to a
    multiple of 8 (< 25%), so a stream of mixed sizes compiles a few shapes."""
    n = max(int(n), 1)
    if n <= 8:
        return 1 << (n - 1).bit_length()
    if n <= 128:
        return -(-n // 8) * 8
    k = math.ceil(4 * math.log2(n / 128))
    while True:
        v = -(-math.ceil(128 * 2 ** (k / 4)) // 8) * 8
        if v >= n:
            return v
        k += 1


def complete_cell(cell, pbc):
    """`cell` (rows: lattice vectors), with its non-periodic rows replaced by an orthonormal complement
    of the periodic ones when it is singular (an ASE cluster's zero cell, a slab without a c vector).
    Only the periodic rows place images; the wrap and the image bound hold for any completion."""
    cell, pbc = np.asarray(cell, float), np.asarray(pbc, bool)
    norms = np.linalg.norm(cell, axis=1)
    if np.prod(norms) > 0 and abs(np.linalg.det(cell)) > 1e-9 * np.prod(norms):
        return cell
    P = cell[pbc]
    if len(P) and np.linalg.matrix_rank(P, tol=1e-9 * norms[pbc].max(initial=0.0)) < len(P):
        raise ValueError(f"the periodic cell vectors are degenerate: {cell[pbc].tolist()}")
    _, _, vt = np.linalg.svd(np.vstack([P, np.zeros((3 - len(P), 3))]))
    out = cell.copy()
    out[~pbc] = vt[len(P):]                              # rows past the rank span the complement
    return out


def image_range(cell, pbc, cutoff):
    """Images per axis, n_k = ceil(cutoff / w_k) on periodic axes (w_k the perpendicular width), 0 on
    the others: with positions wrapped into the cell, every pair within `cutoff` has its image there."""
    vol = abs(np.linalg.det(cell))
    w = [vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)]
    return [math.ceil(cutoff / w[k]) if pbc[k] else 0 for k in range(3)]


@dataclass
class Batch:
    """One padded batch of structures of the same bucket (host arrays).  B structures of N atom slots,
    the live atoms first; S image slots.  Padding structures (index -1) have no live atoms."""
    index: np.ndarray            # (B,) position of each structure in the input; -1 for padding
    n_atoms: np.ndarray          # (B,) int32
    positions: np.ndarray        # (B, N, 3) float64, wrapped into the cell along periodic axes
    species: np.ndarray          # (B, N) int32 model species index (0 on padding)
    live: np.ndarray             # (B, N) bool
    cell: np.ndarray             # (B, 3, 3) float64, completed (`complete_cell`)
    pbc: np.ndarray              # (B, 3) bool
    images: np.ndarray           # (B, S, 3) int32 lattice shifts
    image_mask: np.ndarray       # (B, S) bool
    volume: np.ndarray           # (B,) |det| of the cell as given (0 when singular)

    @property
    def shape(self):
        return self.positions.shape[:2]


def _species_lut(elements):
    lut = np.full(119, -1, np.int32)
    lut[np.asarray(elements, int)] = np.arange(len(elements), dtype=np.int32)
    return lut


def pack(atoms_list, cutoff, elements, max_batch_atoms=None, indices=None, one_batch=False):
    """Padded batches of `atoms_list` (ASE Atoms), grouped by `bucket_size` of the atom count: a list of
    `Batch`, every batch of a bucket with the same B (the last one padded with empty structures, so B does
    not recompile).  `elements`: the model's atomic numbers (the species index order).  `indices`: the
    input positions recorded in `Batch.index` (default 0..len-1).  `cutoff`: the list cutoff the image
    ranges are for (cutoff + skin for MD).  `one_batch=True`: a list of one Batch holding every structure,
    padded to the largest one's bucket, B = len(atoms_list) (the layout for vmapped MD)."""
    cutoff, lut = float(cutoff), _species_lut(elements)
    cap = int(max_batch_atoms or MAX_BATCH_ATOMS)
    indices = list(range(len(atoms_list))) if indices is None else list(indices)
    if one_batch:
        N = bucket_size(max(len(at) for at in atoms_list))
        return [_pack_one(list(zip(indices, atoms_list)), len(atoms_list), N, cutoff, lut)]
    groups = {}
    for i, at in zip(indices, atoms_list):
        groups.setdefault(bucket_size(len(at)), []).append((i, at))
    out = []
    for N, members in sorted(groups.items()):
        B = min(1 << (max(cap // N, 1).bit_length() - 1),          # the largest power of two within cap
                1 << (len(members) - 1).bit_length())              # no more than the bucket needs
        for start in range(0, len(members), B):
            out.append(_pack_one(members[start:start + B], B, N, cutoff, lut))
    return out


def _image_grid(n):
    """The (2 n_x + 1)(2 n_y + 1)(2 n_z + 1) integer lattice shifts of image range n."""
    return np.stack(np.meshgrid(*(np.arange(-k, k + 1) for k in n), indexing="ij"), -1).reshape(-1, 3)


def _pack_one(members, B, N, cutoff, lut):
    """One Batch; vectorised over the structures (per-structure numpy calls cost ~100 us each)."""
    m = len(members)
    pos = np.zeros((B, N, 3))
    num = np.zeros((B, N), np.int64)
    live = np.zeros((B, N), bool)
    cell0 = np.broadcast_to(np.eye(3), (B, 3, 3)).copy()
    pbc = np.zeros((B, 3), bool)
    index = np.full(B, -1, np.int64)
    n_atoms = np.zeros(B, np.int32)
    for b, (i, at) in enumerate(members):              # gather only
        n = len(at)
        pos[b, :n], num[b, :n], live[b, :n] = at.arrays["positions"], at.arrays["numbers"], True
        cell0[b], pbc[b], index[b], n_atoms[b] = at.cell.array, at.pbc, i, n
    for b in np.flatnonzero(~(np.isfinite(pos).all((1, 2)) & np.isfinite(cell0).all((1, 2)))):
        raise ValueError(f"structure {index[b]}: non-finite positions or cell")
    spec = np.where(live, lut[num], 0).astype(np.int32)
    for b in np.flatnonzero((live & (lut[num] < 0)).any(1)):
        raise ValueError(f"structure {index[b]}: elements {sorted(set(num[b][live[b] & (lut[num[b]] < 0)].tolist()))} "
                         "are not in the model")
    volume = np.abs(np.linalg.det(cell0))
    norms = np.prod(np.linalg.norm(cell0, axis=-1), -1)
    cell = cell0.copy()
    for b in np.flatnonzero(~((norms > 0) & (volume > 1e-9 * norms))):
        cell[b] = complete_cell(cell0[b], pbc[b])
    cell[m:] = np.eye(3)
    nrm = np.cross(cell[:, [1, 2, 0]], cell[:, [2, 0, 1]])                # (B, 3, 3): rows c_{k+1} x c_{k+2}
    width = np.abs(np.linalg.det(cell))[:, None] / np.linalg.norm(nrm, axis=-1)
    ranges = np.where(pbc, np.ceil(cutoff / width), 0).astype(int)
    f = np.linalg.solve(np.swapaxes(cell, 1, 2), np.swapaxes(pos, 1, 2))  # fractional, (B, 3, N)
    w = np.where(pbc[:, :, None], np.floor(f), 0.0)
    pos = np.where(live[..., None], pos - np.einsum("bkn,bkc->bnc", w, cell), 0.0)  # integer shifts: exact images
    S = bucket_size(int(np.prod(2 * ranges + 1, axis=1).max()))
    images = np.zeros((B, S, 3), np.int32)
    image_mask = np.zeros((B, S), bool)
    grids = {}
    for b in range(B):
        key = tuple(ranges[b])
        g = grids.get(key)
        if g is None:
            g = grids[key] = _image_grid(key)
        images[b, :len(g)], image_mask[b, :len(g)] = g, True
    return Batch(index, n_atoms, pos, spec, live, cell, pbc, images, image_mask, volume)


def _msn_batched():
    """matscipy-neighbours' batched XLA FFI build (libAtoms/matscipy-neighbours#11), or None."""
    try:
        from matscipy_neighbours import xla
    except ImportError:
        return None
    return getattr(xla, "neighbour_matrix_batched", None)


def neighbours(positions, cell, images, image_mask, live, cutoff, K, park, builder="jax", pbc=None,
               scratch=SCRATCH_BYTES):
    """The dense neighbour lists of a padded batch, on the device, in one call: rij (B, N, K, 3) (masked
    slots parked at `park` along x), idx (B, N, K) (the neighbour's slot in its structure), mask (B, N, K)
    and count (B, N), the true in-cutoff count, which may exceed K (the caller retries with a larger K).
    Pairs with |rij| < cutoff, padding atoms excluded.  Traceable (inside jit).

    builder "jax": all pairs over `images` (B, S, 3) per structure, compacted in order of (j, image) to the
    first K slots (a running count and a scatter: linear in the N S candidates, where top_k sorted them);
    lax.map over structures in steps holding about `scratch` bytes.  "msn": matscipy-
    neighbours' batched FFI (needs `pbc` (B, 3)); its own image search, the same pairs."""
    import jax
    import jax.numpy as jnp
    B, N, _ = positions.shape
    dtype = positions.dtype
    park = jnp.asarray([park, 0.0, 0.0], dtype)
    if builder == "msn":
        n_atoms = live.sum(-1).astype(jnp.int32)
        idx, shift, count, status = _msn_batched()(positions, cell, cutoff, pbc, K, n_atoms=n_atoms)
        mask = jnp.arange(K)[None, None, :] < jnp.minimum(count, K)[..., None]
        rj = jax.vmap(lambda R, i: R[i])(positions, idx)
        rij = rj - positions[:, :, None] + jnp.einsum("bnkc,bcd->bnkd", shift.astype(dtype), cell)
        count = jnp.where(status[:, 1:2] > 0, -1, count)          # invalid input: flagged to the caller
        return jnp.where(mask[..., None], rij, park), jnp.where(mask, idx, 0), mask, count
    S = images.shape[1]
    shifts = jnp.einsum("bsk,bkc->bsc", images.astype(dtype), cell)
    zero = jnp.all(images == 0, -1)
    eye = jnp.eye(N, dtype=bool)[:, :, None]

    def one(a):
        R, sh, sm, z0, lv = a
        d = R[None, :, None] - R[:, None, None] + sh[None, None]             # (i, j, s): r_j - r_i + shift
        ok = (((d * d).sum(-1) < cutoff * cutoff) & lv[:, None, None] & lv[None, :, None] & sm[None, None]
              & ~(eye & z0[None, None]))
        ok = ok.reshape(N, N * S)
        slot = jnp.cumsum(ok, axis=1, dtype=jnp.int32) - 1                  # each candidate's place in its row
        slot = jnp.where(ok & (slot < K), slot, K)                           # K: dropped
        sel = (jnp.zeros((N, K), jnp.int32).at[jnp.arange(N)[:, None], slot]
               .set(jnp.broadcast_to(jnp.arange(N * S, dtype=jnp.int32), (N, N * S)), mode="drop"))
        count = ok.sum(-1, dtype=jnp.int32)
        valid = jnp.arange(K)[None, :] < count[:, None]
        rij = jnp.take_along_axis(d.reshape(N, N * S, 3), sel[..., None], axis=1)
        return jnp.where(valid[..., None], rij, park), jnp.where(valid, sel // S, 0), valid, count

    per = N * N * S * 3 * np.dtype(dtype).itemsize * 4
    bs = int(max(1, min(B, scratch // per)))
    return jax.lax.map(one, (positions, shifts, image_mask, zero, live), batch_size=bs)


def host_neighbours(batch, cutoff, K, park):
    """The batch's dense lists built on the host, one structure at a time (`nlist.dense_graph`:
    matscipy-neighbours' cell list, ASE's without it), as numpy arrays shaped as `neighbours`' output.
    The CPU path until the batched FFI build is released: there the all-pairs JAX builder costs 10-40 us
    per atom at 64-216 atoms, a cell list ~3 (docs/dev/batched-evaluation-results.md)."""
    from .eval.nlist import dense_graph
    B, N = batch.shape
    rij = np.zeros((B, N, K, 3))
    rij[..., 0] = park
    idx = np.zeros((B, N, K), np.int32)
    count = np.zeros((B, N), np.int32)
    for s in np.flatnonzero(batch.index >= 0):
        n, k = int(batch.n_atoms[s]), K
        while True:                                    # dense_graph refuses a row past its capacity
            try:
                g = dense_graph(batch.positions[s, :n], batch.cell[s], batch.pbc[s], cutoff, k,
                                pad_vector=np.array([park, 0.0, 0.0]))
                break
            except ValueError:
                if k >= 1 << 14:
                    raise
                k *= 2
        kk = min(k, K)
        rij[s, :n, :kk], idx[s, :n, :kk], count[s, :n] = np.asarray(g.rij)[:, :kk], np.asarray(g.idx)[:, :kk], g.count
    mask = np.arange(K)[None, None, :] < np.minimum(count, K)[..., None]
    return rij, np.where(mask, idx, 0), mask, count


def efv(model, rij, idx, mask, species, live, chunk=None):
    """Energies (B,), forces (B, N, 3) and virials (B, 3, 3) of a padded batch, the B x N rows evaluated
    as one dense block (`site_energies_dense_blocked`, rows in blocks of `chunk`, None: `default_chunk()`).
    The same quantities as `energy_forces_virial_dense` per structure: E the compensated total of the live
    sites, V = -dE/d(strain) (the symmetric part of -sum rij (x) dE/drij).  Zero on padding atoms."""
    import jax
    import jax.numpy as jnp

    from .eval.edge_model import default_chunk, total_energy
    B, N, K = idx.shape
    rows = B * N
    z = species.reshape(rows)
    zj = jax.vmap(lambda s, i: s[i])(species, idx).reshape(rows, K)
    zi = jnp.broadcast_to(z[:, None], (rows, K))
    m = mask.reshape(rows, K)
    lv = live.reshape(rows)
    chunk = default_chunk() if chunk is None else chunk

    def tot(r):
        e = jnp.where(lv, model.site_energies_dense_blocked(r, zi, zj, m, z, chunk), 0.0).reshape(B, N)
        return jnp.sum(e), e

    r = rij.reshape(rows, K, 3)
    (_, e), g = jax.value_and_grad(tot, has_aux=True)(r)
    E = jax.vmap(total_energy)(e)
    g = jnp.where(m[..., None], g, 0.0)
    nbr = (idx + (jnp.arange(B, dtype=idx.dtype) * N)[:, None, None]).reshape(-1)
    F = (jnp.zeros((rows, 3), g.dtype).at[jnp.arange(rows)].add(g.sum(1)).at[nbr].add(-g.reshape(-1, 3)))
    M = jnp.einsum("bnki,bnkj->bij", rij, g.reshape(B, N, K, 3))
    return E, F.reshape(B, N, 3), -0.5 * (M + jnp.swapaxes(M, 1, 2))


def unpad(batch, x):
    """Per-structure arrays of a padded (B, N, ...) result: a dict input index -> x[b, :n_atoms[b]],
    padding structures dropped."""
    x = np.asarray(x)
    return {int(i): x[b, :int(n)] for b, (i, n) in enumerate(zip(batch.index, batch.n_atoms)) if i >= 0}


# ------------------------------------------------------------------ MD and relaxation with JAX-MD
# ace-jax hosts no integrators or optimizers: JAX-MD's run under jax.vmap over a padded batch, with the
# energy function, neighbour lists and padding helpers below (docs/user/howto/batch.md).

class BatchList(NamedTuple):
    """Neighbour lists of a padded batch for vmapped MD, built at cutoff + skin and reused while every atom
    stays within skin / 2 of `position0`: an edge vector is rij0 + d_j - d_i with d = R - position0, so
    the lists need no minimum image (any cell size) and positions may leave the cell.  Leading axis B;
    under jax.vmap each structure sees its own slice."""
    position0: object            # (B, N, 3) positions at the build, as the integrator holds them
    rij0: object                 # (B, N, K, 3) edge vectors at the build (masked slots parked)
    idx: object                  # (B, N, K) neighbour slot
    mask: object                 # (B, N, K)
    species: object              # (B, N)
    live: object                 # (B, N)


def energy_fn(model, cutoff):
    """`energy(R, neighbor)`: the energy of ONE structure, R (N, 3), neighbor its `BatchList` slice -- the
    form JAX-MD's `simulate.*` and `minimize.*` take (they pass `neighbor=` through), to jax.vmap over the
    batch.  Pairs beyond `cutoff` (the skin pairs) are masked; padding atoms carry no energy and no force.
    Positions-only derivatives of `model`'s dense site energies; the C++ kernel (`eval.ffi.KernelModel`)
    works too: under vmap its rows are flattened into one threaded call.  Its derivatives stop at first
    order in positions (no Hessians, no parameter gradients)."""
    import jax.numpy as jnp

    from .eval.edge_model import total_energy
    park = float(model.pad_cutoff())

    def energy(R, neighbor, **kwargs):
        nl = neighbor
        d = R - nl.position0
        rij = nl.rij0 + d[nl.idx] - d[:, None]
        m = nl.mask & ((rij * rij).sum(-1) < cutoff * cutoff)
        rij = jnp.where(m[..., None], rij, jnp.asarray([park, 0.0, 0.0], rij.dtype))
        z = nl.species
        e = model.site_energies_dense(rij, jnp.broadcast_to(z[:, None], nl.idx.shape), z[nl.idx], m, z)
        return total_energy(jnp.where(nl.live, e, 0.0))
    return energy


def _mass_like(mass, live):
    """mass -- a scalar, one per structure (B,) (a vmapped JAX-MD state's), per atom (..., N), or JAX-MD's
    (..., N, 1) -- as (..., N, 1)."""
    import jax.numpy as jnp
    m = jnp.asarray(mass, jnp.result_type(float))
    if m.ndim == live.ndim + 1:
        m = m[..., 0]
    elif m.ndim < live.ndim and m.shape == live.shape[:m.ndim]:      # leading axes: per structure
        m = m.reshape(m.shape + (1,) * (live.ndim - m.ndim))
    return jnp.broadcast_to(m, live.shape)[..., None]


def initial_momenta(key, live, kT, mass=1.0):
    """Maxwell-Boltzmann momenta (..., N, 3) at kT for the live atoms only, the centre-of-mass momentum
    removed over the live atoms of each structure, zero on padding.  Pass as `momenta=` to a JAX-MD
    `init_fn`: its own draw gives padding atoms momenta and removes the centre of mass over every slot,
    which drifts the live atoms of a padded structure.  `mass`: a scalar or per atom (..., N)."""
    import jax
    import jax.numpy as jnp
    live = jnp.asarray(live)
    m = _mass_like(mass, live)
    p = jnp.sqrt(m * kT) * jax.random.normal(key, live.shape + (3,), m.dtype)
    lv = live[..., None]
    P = jnp.where(lv, p, 0.0).sum(-2, keepdims=True)
    M = jnp.where(lv, m, 0.0).sum(-2, keepdims=True)
    return jnp.where(lv, p - m * P / jnp.maximum(M, 1e-300), 0.0)


def zero_padding(state, live):
    """A JAX-MD state with the padding atoms' momenta set to zero (e.g. after Langevin steps, whose noise
    reaches every slot; padding atoms never interact, so this only cleans what temperatures read)."""
    import jax.numpy as jnp
    return state.set(momentum=jnp.where(jnp.asarray(live)[..., None], state.momentum, 0.0))


def kinetic_energy(momentum, mass, live):
    """Kinetic energy of the live atoms, per structure (sum over the last two axes)."""
    import jax.numpy as jnp
    live = jnp.asarray(live)
    m = _mass_like(mass, live)
    return jnp.where(live[..., None], momentum * momentum / (2 * m), 0.0).sum((-2, -1))


def temperature(momentum, mass, live):
    """kT of the live atoms per structure: 2 KE / (3 n_live), JAX-MD's convention (no centre-of-mass
    correction), but without the padding slots that `jax_md.quantity.temperature` would count."""
    import jax.numpy as jnp
    return 2 * kinetic_energy(momentum, mass, live) / (3 * jnp.maximum(jnp.asarray(live).sum(-1), 1))


def _wrap(positions, cell, pbc):
    """Positions moved into the cell along periodic axes by integer lattice shifts (on the device)."""
    import jax.numpy as jnp
    f = jnp.linalg.solve(jnp.swapaxes(cell, 1, 2), jnp.swapaxes(positions, 1, 2))
    w = jnp.where(pbc[:, :, None], jnp.floor(f), 0.0)
    return positions - jnp.einsum("bkn,bkc->bnc", w, cell)


def _moved(positions, neighbor):
    """(B,) the largest displacement of a live atom since the build."""
    import jax.numpy as jnp
    d = jnp.linalg.norm(positions - neighbor.position0, axis=-1)
    return jnp.where(neighbor.live, d, 0.0).max(-1)


class BatchNeighbours:
    """Neighbour lists for vmapped JAX-MD runs over one padded batch (`BatchCalculator.neighbour_lists`):
    `allocate(R)` builds a `BatchList` at cutoff + skin, `update(R, nl)` rebuilds it when an atom has
    moved skin / 2 (all structures at once: the build is one batched call) and grows K when a structure
    outgrows it (a new shape, so the caller's jitted step recompiles).  Host-side: call between jitted
    chunks of steps."""

    def __init__(self, calc, batch, skin):
        import jax
        import jax.numpy as jnp
        self.calc, self.batch, self.skin = calc, batch, float(skin)
        self.cutoff = calc.cutoff
        self.K = None
        self.rebuilds = 0
        dtype = calc._dtype()
        self.positions = jnp.asarray(batch.positions, dtype)       # wrapped, padded: the initial R
        self.live = jnp.asarray(batch.live)
        self._cell, self._pbc = jnp.asarray(batch.cell, dtype), jnp.asarray(batch.pbc)
        self._wrap, self._moved = jax.jit(_wrap), jax.jit(_moved)

    def allocate(self, positions=None):
        """A BatchList for positions (B, N, 3) (default: the packed ones)."""
        import jax.numpy as jnp
        R = self.positions if positions is None else jnp.asarray(positions, self.positions.dtype)
        W = self._wrap(R, self._cell, self._pbc)
        rij, idx, mask, _, self.K = self.calc._build_lists(self.batch, W, self.cutoff + self.skin, self.K)
        self.rebuilds += 1
        return BatchList(R, rij, idx, mask, jnp.asarray(self.batch.species), self.live)

    def needs_rebuild(self, positions, neighbor):
        """(B,) bool: a live atom has moved more than skin / 2 since the build."""
        return np.asarray(self._moved(positions, neighbor)) > self.skin / 2

    def update(self, positions, neighbor):
        """(neighbor, rebuilt): the same lists while they hold, else lists built at `positions`."""
        if not self.needs_rebuild(positions, neighbor).any():
            return neighbor, False
        return self.allocate(positions), True

    def update_fn(self, builder=None, branchless=None):
        """A jittable `update(R, nl) -> (nl, rebuilt, overflow)` for use INSIDE a compiled loop: when any
        live atom has moved skin / 2 the whole batch's lists are rebuilt at R (lax.cond; one batched build
        call), else nl is returned unchanged.  The shapes are fixed, so a structure that now has more than K
        neighbours is flagged in `overflow` (B,) and its list is clipped: check it after the loop, and on
        overflow re-run from the last good state with `allocate` (which grows K).  `rebuilt`: a scalar
        bool.  builder: "msn" (matscipy-neighbours' batched XLA FFI, when installed) or "jax" (all pairs
        on the device); "host" cannot run inside a trace.  Loop at batch level and vmap only the step:
        `nl, rebuilt, ov = update(state.position, nl); state = jax.vmap(apply)(state, neighbor=nl)`.

        On a GPU, XLA reads a lax.cond predicate back to the host: one small device-to-host copy and
        a stream sync per call (measured with nsys; XLA's command-buffer conditionals did not remove it).
        `branchless=True` avoids it: the batch is built on every call and kept per structure where an atom
        moved (jnp.where), so the loop never leaves the device, at the cost of a build per call.  None (the
        default): branch-free on a GPU, lax.cond on a CPU (no copy there; a build per step would cost
        ~30%).  Cheapest on either: call `update` every n steps (an outer loop over inner fori_loops), one
        check per n steps.  lestrade RTX 4000 Ada (shared), 5 small cells, 200 NVE steps: lists fixed 62
        ms, cond every step 485 ms, branch-free 100-145 ms, cond every 10 steps 78 ms.  Branch-free
        rebuilds only the structures that moved."""
        import jax
        import jax.numpy as jnp
        builder = builder or ("msn" if _msn_batched() is not None else "jax")
        if builder not in ("msn", "jax"):
            raise ValueError(f"update_fn needs a traceable builder, 'msn' or 'jax', got {builder!r}")
        if self.K is None:
            raise ValueError("call allocate() first: it fixes K")
        if branchless is None:
            branchless = jax.default_backend() != "cpu"
        b, K, rc, half = self.batch, self.K, self.cutoff + self.skin, self.skin / 2
        cell, pbc = self._cell, self._pbc
        images, image_mask = jnp.asarray(b.images), jnp.asarray(b.image_mask)
        park = float(self.calc.eval_model.pad_cutoff())

        def build(R, live):
            rij, idx, mask, count = neighbours(_wrap(R, cell, pbc), cell, images, image_mask, live, rc, K, park,
                                               builder, pbc)
            return rij, idx, mask, count.max(-1) > K

        def update(R, nl):
            moved = _moved(R, nl) > half
            if branchless:                    # build every call, keep it per structure where an atom moved
                rij, idx, mask, over = build(R, nl.live)
                sel = lambda new, old: jnp.where(moved.reshape(moved.shape + (1,) * (new.ndim - 1)), new, old)
                nl = nl._replace(position0=sel(R, nl.position0), rij0=sel(rij, nl.rij0), idx=sel(idx, nl.idx),
                                 mask=sel(mask, nl.mask))
                return nl, moved.any(), moved & over

            def rebuild(_):
                rij, idx, mask, over = build(R, nl.live)
                return nl._replace(position0=R, rij0=rij, idx=idx, mask=mask), over

            rebuilt = moved.any()
            nl, overflow = jax.lax.cond(rebuilt, rebuild, lambda _: (nl, jnp.zeros(R.shape[0], bool)), None)
            return nl, rebuilt, overflow
        return update

    def unpad(self, x):
        """Per-structure arrays of a padded (B, N, ...) result, in input order (a list)."""
        d = unpad(self.batch, x)
        return [d[i] for i in sorted(d)]


def _build(positions, cell, images, image_mask, live, pbc, cutoff, K, park, builder):
    rij, idx, mask, count = neighbours(positions, cell, images, image_mask, live, cutoff, K, park, builder, pbc)
    return rij, idx, mask, count.max(-1, initial=0)


def _efv_slots(model, rij, idx, mask, species, live, K):
    """`efv` on the first K slots: the lists are compacted, so a K at the batch's largest count drops
    only empty slots (the dense model's work is proportional to K)."""
    return efv(model, rij[:, :, :K], idx[:, :, :K], mask[:, :, :K], species, live)


@dataclass
class BatchResults:
    """`BatchCalculator.calculate`'s results, in input order."""
    energy: np.ndarray           # (n,) eV (relative to E0 for energy_reference="E0")
    forces: list                 # n arrays (n_i, 3), eV/A
    virial: np.ndarray           # (n, 3, 3) eV; NaN where a structure's cell volume is 0 and it went
    #                              through ACECalculator (above N_DEVICE)
    stress: np.ndarray           # (n, 6) eV/A^3, ASE Voigt order; NaN rows unless fully periodic
    e0_offset: np.ndarray        # (n,) sum_i E0[z_i] for energy_reference="E0", else 0
    timing: dict

    def to_atoms(self, atoms_list):
        """Copies of `atoms_list` with these results attached (SinglePointCalculator)."""
        from ase.calculators.singlepoint import SinglePointCalculator
        out = []
        for k, at in enumerate(atoms_list):
            c = at.copy()
            kw = dict(energy=float(self.energy[k]), free_energy=float(self.energy[k]), forces=self.forces[k])
            if np.isfinite(self.stress[k]).all():
                kw["stress"] = self.stress[k]
            c.calc = SinglePointCalculator(c, **kw)
            out.append(c)
        return out


class BatchCalculator:
    """Energies, forces and stresses of many structures, batched: `BatchCalculator(model).calculate(
    atoms_list)` -> `BatchResults`.

    The evaluation model is resolved exactly as `ACECalculator`'s (one is built inside, `calc`): `model`,
    `meta`, `cutoff`, `dtype`, `lean`, `spline_tol`, `spline_intervals`, `radial_table`, `aa_dag`, `ffi`,
    `ffi_threads` and `energy_reference` mean the same, and the same evaluation model is used.  Results
    agree with `ACECalculator` structure by structure to roundoff.

    Structures up to `n_device` atoms are padded into buckets (`pack`) and evaluated `max_batch_atoms`
    padded atoms per call; larger ones go through `calc` one at a time (`n_device` None: `N_DEVICE`, or
    `N_DEVICE_CPU_JAX` for the JAX model on a CPU, where batching gains only on the smallest cells).  `builder` picks the neighbour
    lists: "msn" (matscipy-neighbours' batched XLA FFI, one call per batch, CPU or CUDA), "jax" (all
    pairs over images on the device, `neighbours`: fine on a GPU, 10-40 us per atom on a CPU), "host"
    (a cell list per structure on the host, `host_neighbours`); "auto" takes "msn" when installed, else
    "jax" on a GPU and, on a CPU, per batch "jax" up to `CPU_ALLPAIRS_MAX` candidates per atom, else
    "host".  Each (bucket, B, K, S) compiles once; a structure with more than K neighbours re-runs its batch
    at a larger K (a recompile), never truncating.  Positions-only derivatives (forces, virial)."""

    def __init__(self, model, meta=None, cutoff=None, dtype=None, lean=True, spline_tol=AUTO,
                 spline_intervals=None, radial_table=None, aa_dag="auto", ffi=False, ffi_threads=0,
                 energy_reference="absolute", max_batch_atoms=None, n_device=None, builder="auto"):
        from .calc.point import ACECalculator
        if builder not in ("auto", "jax", "msn", "host"):
            raise ValueError(f"builder must be 'auto', 'jax', 'msn' or 'host', got {builder!r}")
        if builder == "msn" and _msn_batched() is None:
            raise ValueError("builder='msn' needs matscipy-neighbours with the batched XLA FFI build "
                             "(matscipy_neighbours.xla.neighbour_matrix_batched)")
        self.calc = ACECalculator(model, meta=meta, cutoff=cutoff, dtype=dtype, layout="dense", skin=0.0,
                                  lean=lean, spline_tol=spline_tol, spline_intervals=spline_intervals,
                                  radial_table=radial_table, aa_dag=aa_dag, ffi=ffi, ffi_threads=ffi_threads,
                                  energy_reference=energy_reference)
        import jax
        cpu = jax.default_backend() == "cpu"
        if builder == "auto" and (_msn_batched() is not None or not cpu):
            builder = "msn" if _msn_batched() is not None else "jax"
        self.builder = builder           # "auto" here: per batch on a CPU (`_builder_for`)
        self.max_batch_atoms = int(max_batch_atoms or MAX_BATCH_ATOMS)
        self.n_device = int(n_device if n_device is not None
                            else N_DEVICE_CPU_JAX if cpu and self.calc.ffi is None else N_DEVICE)
        self._K = {}                     # bucket N -> neighbour slots, grown on overflow
        self._shapes = set()             # compiled (N, B, K, S)
        import equinox as eqx
        self._jit_build, self._jit_efv = eqx.filter_jit(_build), eqx.filter_jit(_efv_slots)

    # the evaluation options, as ACECalculator reports them
    meta = property(lambda self: self.calc.meta)
    cutoff = property(lambda self: self.calc.cutoff)
    model = property(lambda self: self.calc.model)
    eval_model = property(lambda self: self.calc.eval_model)
    ffi = property(lambda self: self.calc.ffi)
    ffi_reason = property(lambda self: self.calc.ffi_reason)
    energy_reference = property(lambda self: self.calc.energy_reference)

    def _builder_for(self, batch):
        if self.builder != "auto":
            return self.builder
        N, S = batch.positions.shape[1], batch.images.shape[1]
        return "jax" if N * S <= CPU_ALLPAIRS_MAX else "host"

    def _dtype(self):
        import jax.numpy as jnp
        return np.dtype(self.calc.dtype or jnp.zeros(()).dtype)

    def _k_guess(self, batch, rc):
        """Neighbour slots for a first call on a bucket: the densest periodic structure's expected count
        within rc, with headroom (an overflow retries), capped at the candidates."""
        from .calc.skin import round_k
        est = []
        for b in np.flatnonzero(batch.index >= 0):
            n = int(batch.n_atoms[b])
            if batch.pbc[b].all() and batch.volume[b] > 0:
                est.append(1.3 * 4 / 3 * math.pi * rc ** 3 * n / batch.volume[b] + 8)
            else:
                est.append(n)
        N, S = batch.positions.shape[1], batch.images.shape[1]
        return round_k(min(max(est, default=1), N * S))

    def _build_lists(self, b, positions, cutoff, K, builder=None):
        """(rij, idx, mask, kmax, K): batch b's dense lists at `cutoff` for `positions` (B, N, 3, on the
        device, wrapped into the cells), re-built at a larger K until no structure overflows (K None: a
        guess from the density).  kmax (B,) host: each structure's largest count."""
        import jax
        from .calc.skin import round_k
        builder = builder or self._builder_for(b)
        stats = getattr(self, "_stats", None)
        if stats is None:
            stats = self._stats = {"compiles": 0, "k_retries": 0}
        K = K or self._k_guess(b, cutoff)
        park = float(self.eval_model.pad_cutoff())
        (B, N), S = b.shape, b.images.shape[1]
        while True:
            key = ("build", builder, N, B, K, S if builder == "jax" else 0, cutoff)
            stats["compiles"] += key not in self._shapes
            self._shapes.add(key)
            if builder == "host":
                rij, idx, mask, count = host_neighbours(dataclasses.replace(b, positions=np.asarray(positions)),
                                                        cutoff, K, park)
                rij, kmax = jax.numpy.asarray(rij, positions.dtype), count.max(-1, initial=0)
            else:
                rij, idx, mask, kmax = self._jit_build(positions, jax.numpy.asarray(b.cell, positions.dtype),
                                                       b.images, b.image_mask, b.live, b.pbc, cutoff, K, park,
                                                       builder)
                kmax = np.asarray(kmax)
            if (kmax < 0).any():
                bad = [int(i) for i in b.index[kmax < 0]]
                raise ValueError(f"structures {bad}: invalid input (non-finite positions or a degenerate cell)")
            if kmax.max(initial=0) <= K:
                return rij, idx, mask, kmax, K
            K = round_k(int(kmax.max()))
            stats["k_retries"] += 1

    def energy_fn(self):
        """JAX-MD's energy function for one structure, `energy(R, neighbor=...)`, over this calculator's
        evaluation model (`batch.energy_fn`); jax.vmap the integrator or optimizer over the batch.  With
        ffi the C++ kernel evaluates it: one threaded call for the whole vmapped batch."""
        return energy_fn(self.eval_model, self.cutoff)

    def neighbour_lists(self, atoms_list, skin=1.0):
        """`BatchNeighbours` for vmapped MD or relaxation of `atoms_list`: all structures in ONE padded batch
        (B = len(atoms_list), N the largest one's bucket), lists at cutoff + skin.  Its `positions` are the
        initial R (B, N, 3) and `live` the (B, N) mask; `unpad` returns per-structure arrays."""
        if not skin > 0:
            raise ValueError(f"skin must be > 0, got {skin!r}")
        (b,) = pack(atoms_list, self.cutoff + skin, self.meta["elements"], one_batch=True)
        return BatchNeighbours(self, b, skin)

    def calculate(self, atoms_list):
        """Energies, forces and stresses of `atoms_list` (ASE Atoms): a `BatchResults` in input order."""
        import jax
        from .calc.skin import round_k
        from .eval.model import highest_precision
        t0 = time.perf_counter()
        n = len(atoms_list)
        small = [i for i, at in enumerate(atoms_list) if len(at) <= self.n_device]
        large = [i for i, at in enumerate(atoms_list) if len(at) > self.n_device]
        energy, virial = np.full(n, np.nan), np.full((n, 3, 3), np.nan)
        forces = [None] * n
        e0 = np.asarray(self.calc._e0, np.float64)
        e0_offset = np.array([math.fsum(e0[self.calc._species_index(at.get_atomic_numbers())])
                              if self.energy_reference == "E0" else 0.0 for at in atoms_list])
        batches = pack([atoms_list[i] for i in small], self.cutoff, self.meta["elements"],
                       self.max_batch_atoms, indices=small)
        t_pack = time.perf_counter() - t0
        dtype, model = self._dtype(), self.eval_model
        self._stats = {"compiles": 0, "k_retries": 0}
        buckets, builders = {}, {}
        t1 = time.perf_counter()
        with highest_precision():
            for b in batches:
                N = b.positions.shape[1]
                B = b.positions.shape[0]
                buckets[N] = buckets.get(N, 0) + 1
                builder = self._builder_for(b)
                builders[builder] = builders.get(builder, 0) + 1
                rij, idx, mask, kmax, self._K[N] = self._build_lists(
                    b, jax.numpy.asarray(b.positions, dtype), self.cutoff, self._K.get(N), builder)
                Ke = min(self._K[N], round_k(int(kmax.max(initial=1))))   # evaluate the occupied slots only
                self._stats["compiles"] += ("efv", N, B, Ke) not in self._shapes
                self._shapes.add(("efv", N, B, Ke))
                E, F, V = jax.block_until_ready(self._jit_efv(model, rij, idx, mask, b.species, b.live, Ke))
                E, V = np.asarray(E, np.float64), np.asarray(V, np.float64)
                for i, f in unpad(b, F).items():
                    forces[i] = np.asarray(f, np.float64)
                live = b.index >= 0
                energy[b.index[live]], virial[b.index[live]] = E[live], V[live]
        t_eval = time.perf_counter() - t1
        t2 = time.perf_counter()
        for i in large:                                  # one at a time: the per-structure list paths
            at = atoms_list[i].copy()
            at.calc = self.calc
            energy[i], forces[i] = at.get_potential_energy(), np.asarray(at.get_forces())
            vol = abs(np.linalg.det(at.get_cell().array))
            if vol > 0:
                virial[i] = -at.get_stress(voigt=False) * vol
        vol = np.abs(np.linalg.det(np.array([at.cell.array for at in atoms_list]).reshape(n, 3, 3)))
        ok = np.array([at.pbc.all() for at in atoms_list], bool) & (vol > 0)
        s = -virial / np.where(ok, vol, np.nan)[:, None, None]
        stress = np.where(ok[:, None], s[:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]], np.nan)   # ASE Voigt
        timing = {"pack_s": t_pack, "batched_s": t_eval, "single_s": time.perf_counter() - t2,
                  "batches": len(batches), "buckets": buckets, **self._stats,
                  "single": len(large), "builder": builders}
        self.last_timing = timing
        return BatchResults(energy, forces, virial, stress, e0_offset, timing)
