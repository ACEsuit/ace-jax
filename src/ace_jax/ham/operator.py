"""From predicted blocks to H(k), S(k) and eigenvalues, on the device.

`HamiltonianPredictor` runs the on-site and off-site models on one set of site features and
keeps the blocks where they were computed.  `hk` sums them into the Bloch matrices
M(k) = sum_R exp(2 pi i k.R) M(R) (k in fractional coordinates of the reciprocal cell) and
`bands` solves H(k) c = e S(k) c through a Cholesky factor of S(k).  Nothing the size of the
blocks crosses to the host: the lists go up (a few MB), the eigenvalues come down."""
import jax
import jax.numpy as jnp
import numpy as np

from ..eval.nlist import dense_graph, have_matscipy_neighbours
from ..operators import bond as B
from ..operators import offsite_fast as OF
from ..operators.offsite_fast import OffsitePredictor
from ..operators.onsite import OnsitePredictor, _features, _readout, site_graph


def _scatter(M, blocks, rows, cols, weight, n_orb):
    """Add `blocks` (..., n_a, n_b), each times its `weight` (...,), into the flat (n_orb^2,)
    matrix `M` at row offsets `rows` and column offsets `cols` (...,)."""
    na, nb = blocks.shape[-2:]
    idx = ((rows[..., None, None] + jnp.arange(na)[:, None]) * n_orb + cols[..., None, None] + jnp.arange(nb))
    return M.at[idx.reshape(-1)].add((blocks * weight[..., None, None]).reshape(-1))


def _bloch_impl(on, off, lists, k, channel, n_orb):
    """(n_orb, n_orb) complex: the Bloch sum of one operator channel at fractional k."""
    cdtype = jnp.result_type(k.dtype, jnp.complex64)
    M = jnp.zeros(n_orb * n_orb, cdtype)
    for blocks, rows in zip(on, lists["on_rows"]):
        M = _scatter(M, blocks[:, channel].astype(cdtype), rows, rows, jnp.ones(rows.shape, cdtype), n_orb)
    for key, blocks in off.items():
        rows, cols, R, valid = lists["off"][key]                               # (n_a, K_ab) each
        w = jnp.exp(2j * jnp.pi * (R @ k)).astype(cdtype) * valid
        M = _scatter(M, blocks[:, :, channel].astype(cdtype), rows, cols, w, n_orb)
    return M.reshape(n_orb, n_orb)


_bloch = jax.jit(_bloch_impl, static_argnums=(4, 5))


def _eigvals_impl(H, S):
    """Eigenvalues of H c = e S c, both Hermitian and S positive definite."""
    H, S = 0.5 * (H + H.conj().T), 0.5 * (S + S.conj().T)
    L = jnp.linalg.cholesky(S)
    X = jax.scipy.linalg.solve_triangular(L, H, lower=True)
    Ht = jax.scipy.linalg.solve_triangular(L, X.conj().T, lower=True).conj().T
    return jnp.linalg.eigvalsh(0.5 * (Ht + Ht.conj().T))


_eigvals = jax.jit(_eigvals_impl)
_eigvals_k = jax.jit(jax.vmap(_eigvals_impl))             # a batch of k-points in one call


def _bloch_blocks_impl(B, idx, R, kpts, n_orb):
    """(n_k, n_orb, n_orb): the Bloch sums of stacked blocks at every k of `kpts` at once.  B: one
    (m, n_a, n_b) array per block shape, idx: their flat positions (m, n_a, n_b) in the n_orb^2
    matrix, R: their cells (m, 3)."""
    cdtype = jnp.result_type(kpts.dtype, jnp.complex64)

    def one(k):
        M = jnp.zeros(n_orb * n_orb, cdtype)
        for b, i, r in zip(B, idx, R):
            M = M.at[i.reshape(-1)].add((jnp.exp(2j * jnp.pi * (r @ k)).astype(cdtype)[:, None, None] * b).reshape(-1))
        return M.reshape(n_orb, n_orb)

    return jax.vmap(one)(kpts)


_bloch_blocks = jax.jit(_bloch_blocks_impl, static_argnums=4)


class BlockSum:
    """The Bloch sums of a dictionary of blocks {(Rx, Ry, Rz, i, j): (n_i, n_j)} (reference data, or
    any code's predictions) on the device, every k-point of a batch in one compiled call.  The
    index lists are built once; `matrices(blocks, kpts)` then takes any blocks with these keys."""

    def __init__(self, keys, numbers, basis):
        numbers = np.asarray(numbers)
        norb = np.array([basis.n_orb(int(z)) for z in numbers])
        off = np.concatenate([[0], np.cumsum(norb)])
        self.n_orb = int(off[-1])
        groups = {}
        for key in keys:
            groups.setdefault((int(norb[key[3]]), int(norb[key[4]])), []).append(key)
        self.groups = [(shape, ks) for shape, ks in sorted(groups.items())]
        self.idx, self.R = [], []
        for (na, nb), ks in self.groups:
            a = np.asarray(ks)
            rows, cols = off[a[:, 3]], off[a[:, 4]]
            self.idx.append(jnp.asarray((rows[:, None, None] + np.arange(na)[:, None]) * self.n_orb + cols[:, None, None] + np.arange(nb), jnp.int32))
            self.R.append(jnp.asarray(a[:, :3], float))

    def matrices(self, blocks, kpts, chunk=64):
        """(n_k, n_orb, n_orb) complex Hermitian Bloch matrices at the fractional k-points."""
        B = [jnp.asarray(np.stack([blocks[k] for k in ks])) for _, ks in self.groups]
        kpts = jnp.asarray(np.atleast_2d(kpts), B[0].dtype if B else float)
        out = [_bloch_blocks(B, self.idx, self.R, kpts[i:i + chunk], self.n_orb) for i in range(0, len(kpts), chunk)]
        M = jnp.concatenate(out)
        return 0.5 * (M + jnp.swapaxes(M.conj(), 1, 2))


def block_bands(bs, H, S, kpts, chunk=48):
    """(n_k, n_orb) eigenvalues, as a host array, of the generalised problem of two dictionaries of
    blocks at the fractional k-points: Bloch sums and eigensolves of `chunk` k-points at a time on
    the device, so that only that many matrices exist at once."""
    kpts = np.atleast_2d(np.asarray(kpts, float))
    Hb = [jnp.asarray(np.stack([H[k] for k in ks])) for _, ks in bs.groups]
    Sb = [jnp.asarray(np.stack([S[k] for k in ks])) for _, ks in bs.groups]
    out = []
    for i in range(0, len(kpts), chunk):
        k = jnp.asarray(kpts[i:i + chunk], Hb[0].dtype)
        out.append(np.asarray(_eigvals_k(_bloch_blocks(Hb, bs.idx, bs.R, k, bs.n_orb), _bloch_blocks(Sb, bs.idx, bs.R, k, bs.n_orb))))
    return np.concatenate(out)


def eigvals(H, S, chunk=64):
    """(n_k, n_orb) eigenvalues of H(k) c = e S(k) c for batches of matrices, on the device."""
    return jnp.concatenate([_eigvals_k(H[i:i + chunk], S[i:i + chunk]) for i in range(0, len(H), chunk)])


def _device_impl(site, readout, packs, pos, cell, species, site_species, s_rij, s_idx, s_count, b_rij, b_idx, b_count, env, static):
    """Everything `prepare` does after the neighbour lists, on the device: site features, on-site
    blocks, the reverse slot and image of every bond, off-site blocks, and the index lists of the
    Bloch sum.  `species`: index into the orbital basis' species, per atom; `site_species`: index
    into the site basis' elements (its radials are tabulated in that order)."""
    args, counts, norbs, rcut_bond, zs, kpair = static
    n, Ks = s_idx.shape
    K = b_idx.shape[1]
    dtype = b_rij.dtype
    # site features from the dense site list
    live = (jnp.arange(Ks)[None, :] < s_count[:, None]).reshape(-1)
    centre = jnp.repeat(jnp.arange(n), Ks)
    F = site.features(s_rij.reshape(-1, 3), site_species[centre], site_species[s_idx.reshape(-1)], centre, n, live)
    atoms = tuple(jnp.nonzero(species == k, size=c)[0] for k, c in enumerate(counts))
    on = tuple(readout.blocks(F, k, a) for k, a in enumerate(atoms))
    # bonds: (atom, slot) is the dense bond list itself
    mask = jnp.arange(K)[None, :] < b_count[:, None]
    i = jnp.broadcast_to(jnp.arange(n)[:, None], (n, K))
    R = jnp.rint((b_rij - (pos[b_idx] - pos[:, None, :])) @ jnp.linalg.inv(cell)).astype(jnp.int32)
    R = jnp.where(mask[..., None], R, 0)
    # reverse slot: the bond (j, i, -R).  Sort both key lists; padding (key n) sorts last in both.
    fi, fj, fR = jnp.where(mask, i, n).reshape(-1), jnp.where(mask, b_idx, n).reshape(-1), R.reshape(-1, 3)
    fwd = jnp.lexsort((fR[:, 2], fR[:, 1], fR[:, 0], fj, fi))
    bwd = jnp.lexsort((-fR[:, 2], -fR[:, 1], -fR[:, 0], fi, fj))
    rev = jnp.zeros(n * K, jnp.int32).at[bwd].set(fwd.astype(jnp.int32)).reshape(n, K)
    norb = jnp.asarray(norbs, jnp.int32)[species]
    offsets = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(norb)[:-1]])
    # one dense layout per ordered species pair: the rows are the atoms of the first species, and
    # their neighbours of the second are packed into the first slots (a stable sort per row)
    zj = species[b_idx]
    where = jnp.full(n * K + 1, 0, jnp.int32)             # global (atom, slot) -> position in its pair's layout
    part = {}
    for ka, za in enumerate(zs):
        if not counts[ka]:
            continue
        rows = atoms[ka]
        for kb, zb in enumerate(zs):
            if not counts[kb]:
                continue
            Kab = kpair[ka][kb]
            want = mask[rows] & (zj[rows] == kb)
            order = jnp.argsort(~want, axis=1, stable=True)[:, :Kab]              # (n_a, K_ab) slots of the bond list
            m = jnp.take_along_axis(want, order, axis=1)
            gflat = rows[:, None] * K + order
            where = where.at[jnp.where(m, gflat, n * K).reshape(-1)].set(jnp.arange(m.size, dtype=jnp.int32))
            part[(za, zb)] = (rows, order, m, gflat)
    lay, lists_off = {}, {}
    pad = jnp.asarray([rcut_bond, 0.0, 0.0], dtype)
    for key, (rows, order, m, gflat) in part.items():
        rvec = jnp.where(m[..., None], jnp.take_along_axis(b_rij[rows], order[..., None], axis=1), pad)
        lay[key] = (rows, rvec, m.astype(dtype), where[rev.reshape(-1)[gflat.reshape(-1)]])
        nbr = jnp.take_along_axis(b_idx[rows], order, axis=1)
        lists_off[key] = (jnp.broadcast_to(offsets[rows][:, None], m.shape), offsets[nbr],
                          jnp.take_along_axis(R[rows], order[..., None], axis=1).astype(dtype), m.astype(jnp.float32))
    off = OF._blocks_impl(packs, F, lay, rcut_bond, args, env)
    lists = {"off": lists_off, "on_rows": tuple(offsets[a] for a in atoms)}
    ok = jnp.all(jnp.where(mask, (b_idx.reshape(-1)[rev.reshape(-1)].reshape(n, K) == i), True))
    return on, off, lists, ok


_device = jax.jit(_device_impl, static_argnums=14)


# compiled, with the slot count static: an eager slice uploads its bounds as scalars
@jax.jit
def _pair_slots(species, b_idx, b_count, ns):
    """(ns, ns): the largest number of neighbours of species b any atom of species a has."""
    mask = jnp.arange(b_idx.shape[1])[None, :] < b_count[:, None]
    per = jax.vmap(lambda kb: jnp.sum(mask & (species[b_idx] == kb), axis=1))(ns)        # (ns, n)
    return jax.vmap(lambda ka: jnp.max(jnp.where(species[None, :] == ka, per, 0), axis=1))(ns)


_take_species = jax.jit(lambda species, idx: species[idx])        # compiled: an eager gather uploads its bounds
_trim = jax.jit(lambda rij, idx, kt: (rij[:, :kt], idx[:, :kt]), static_argnums=2)


class Prepared:
    """One structure, ready for `hk` and `bands`: the blocks on the device and the index lists
    that place them in the orbital matrix (`n_orb` orbitals; atom a starts at `offsets[a]`)."""

    def __init__(self, on, off, lists, n_orb, offsets, graph):
        self.on, self.off, self.lists, self.n_orb, self.offsets, self.graph = on, off, lists, n_orb, offsets, graph


class HamiltonianPredictor:
    """`onsite` and `offsite`: fitted models on the same site basis, orbital basis and channels."""

    def __init__(self, onsite, offsite, threads=1):
        if onsite.channels != offsite.channels or onsite.orbitals != offsite.orbitals:
            raise ValueError("the on-site and off-site models have different channels or orbital bases")
        if onsite.basis is not offsite.bond.site:
            raise ValueError("the on-site and off-site models must share one site basis (the same object): "
                             "its features are evaluated once for both")
        self.onsite, self.offsite = OnsitePredictor(onsite), OffsitePredictor(offsite, threads=threads)
        self.channels, self.orbitals = onsite.channels, onsite.orbitals
        self._k = {}                                   # slots of the device neighbour lists, by cutoff
        self._ns = None

    def _species_range(self):
        if self._ns is None:
            self._ns = jax.device_put(np.arange(len(self.orbitals.species), dtype=np.int32))
        return self._ns

    def _dense(self, positions, cell, pbc, cutoff):
        """Dense neighbour list on the GPU (matscipy-neighbours' CUDA backend), growing the slot
        count until every atom fits."""
        K = self._k.get(cutoff, 64)
        while True:
            try:
                g = dense_graph(positions, cell, pbc, cutoff, K, device="cuda")
                self._k[cutoff] = K
                # keep only the occupied slots, in steps of 16 so that small changes of the largest
                # neighbour count do not recompile (one scalar comes down to the host for this)
                kt = min(K, max(16, -(-int(g.count.max()) // 16) * 16))
                rij, idx = _trim(g.rij, g.idx, kt)
                return g._replace(rij=rij, idx=idx)
            except ValueError as e:
                if "max_neighbours" not in str(e) or K > 1 << 14:
                    raise
                K *= 2

    def prepare_device(self, positions, cell, numbers, pbc=True):
        """`prepare` with the neighbour lists built on the GPU and every list derived from them
        there: the only uploads are the positions, the cell and the species.  Needs a CUDA
        backend and matscipy-neighbours built with CUDA (`eval.nlist.dense_graph(device="cuda")`)."""
        if jax.default_backend() != "gpu" or not have_matscipy_neighbours():
            raise RuntimeError("prepare_device needs JAX on a GPU and matscipy-neighbours built with CUDA")
        orb, basis = self.orbitals, self.offsite.model.bond
        numbers = np.asarray(numbers)
        species = np.asarray([orb.species.index(int(z)) for z in numbers], np.int32) if all(
            int(z) in orb.species for z in numbers) else None
        if species is None:
            raise KeyError("the structure has a species without an orbital basis")
        gs = self._dense(positions, cell, pbc, basis.site.rcut)
        gb = self._dense(positions, cell, pbc, basis.rcut_bond)
        dtype = gb.rij.dtype
        counts = tuple(int((species == k).sum()) for k in range(len(orb.species)))
        norbs = tuple(orb.n_orb(z) for z in orb.species)
        pos_d, cell_d, sp_d, ss_d = (jax.device_put(np.asarray(positions, dtype)), jax.device_put(np.asarray(cell, dtype)),
                                     jax.device_put(species), jax.device_put(basis.site.species_index(numbers)))
        # slots per species pair, in steps of 8 (a small array comes down to the host for this)
        kp = np.asarray(_pair_slots(sp_d, gb.idx, gb.count, self._species_range()))
        kpair = tuple(tuple(int(min(gb.idx.shape[1], max(8, -(-int(v) // 8) * 8))) for v in row) for row in kp)
        static = (self.offsite.static(), counts, norbs, basis.rcut_bond, tuple(orb.species), kpair)
        env = None
        if basis.n_env:
            ge = self._dense(positions, cell, pbc, basis.r_env)
            env = (ge.rij, ge.count, _take_species(ss_d, ge.idx))
        on, off, lists, ok = _device(basis.site, self.onsite.readout, self.offsite.packs, pos_d, cell_d, sp_d, ss_d,
                                     gs.rij, gs.idx, gs.count, gb.rij, gb.idx, gb.count, env, static)
        prep = Prepared(on, off, lists, int(sum(orb.n_orb(z) for z in numbers)), None, None)
        prep.reverse_ok = ok                              # device scalar: every bond found its reverse
        return prep

    def prepare(self, positions, cell, numbers, pbc=True):
        """Evaluate every block of the structure; they stay on the device."""
        orb, basis = self.orbitals, self.offsite.model.bond
        numbers = np.asarray(numbers)
        n = len(numbers)
        for z in set(int(z) for z in numbers):
            orb.ls(z)
        rij, zi, zj, seg, _ = site_graph(basis.site, positions, cell, numbers, pbc)
        if len(rij) == 0:
            z1 = np.zeros(1, np.int32)
            rij, zi, zj, seg = np.array([[basis.site.rcut, 0.0, 0.0]]), z1, z1, z1
        F = _features(basis.site, rij, zi, zj, seg, n)
        atoms = tuple(np.flatnonzero(numbers == z).astype(np.int32) for z in orb.species)
        on = _readout(self.onsite.readout, F, atoms)
        g = B.bond_graph(basis, positions, cell, numbers, pbc)
        pl = OF.PairLayouts(g, numbers, orb.species, basis.rcut_bond)
        off = self.offsite.blocks(F, pl.lay, B.env_lists(basis, positions, cell, pbc, numbers))
        offsets = np.concatenate([[0], np.cumsum([orb.n_orb(z) for z in numbers])])
        lists_off = {}
        for key in off:
            b = pl.bond[key]                                                  # (n_a, K_ab) bond per slot, -1 padding
            live, e = b >= 0, np.maximum(b, 0)
            lists_off[key] = (jnp.asarray(offsets[g.i[e]], jnp.int32), jnp.asarray(offsets[g.j[e]], jnp.int32),
                              jnp.asarray(np.where(live[..., None], g.R[e], 0), off[key].dtype), jnp.asarray(live.astype(np.float32)))
        lists = {"off": lists_off, "on_rows": tuple(jnp.asarray(offsets[a], jnp.int32) for a in atoms)}
        prep = Prepared(on, off, lists, int(offsets[-1]), offsets, g)
        prep.inputs = (F, atoms, pl.lay, B.env_lists(basis, positions, cell, pbc, numbers))    # what `spectrum_fn` re-evaluates from
        return prep

    def spectrum_fn(self, prep, kpts, overlap="S", channel="H", overlap_shift=0.0):
        """(fn, params) for learning on spectral observations: fn(params) -> (n_k, n_orb) eigenvalues
        of the structure of `prep` (from `prepare`) at the fractional k-points, one compiled program
        from the coefficients to the eigenvalues that `jax.grad` differentiates.  params: the
        floating-point arrays of the on-site readout and the off-site packs (the fitted
        coefficients in the layout the predictors evaluate; the index arrays stay fixed), a pytree
        to hand to an optimiser; `fn(params)` at the returned params is `bands`.

        The site features, neighbour lists and index lists are those of `prep`: constants of fn.
        overlap=None: an orthonormal basis.  overlap_shift: added to the diagonal of S(k) (for
        models whose predicted overlap is not positive definite, e.g. random ones in tests).
        Eigenvalue gradients are those of `jnp.linalg.eigh`: exact degeneracies can give NaN."""
        import equinox as eqx
        b = self.offsite.model.bond
        params, static = eqx.partition((self.onsite.readout, self.offsite.packs), eqx.is_inexact_array)
        F, atoms, lay, env = prep.inputs
        lay = {k: tuple(jnp.asarray(x) for x in v) for k, v in lay.items()}
        env = None if env is None or not b.n_env else tuple(jnp.asarray(x) for x in env)
        args, lists, n_orb = self.offsite.static(), prep.lists, prep.n_orb
        kpts = jnp.asarray(np.atleast_2d(np.asarray(kpts, float)), prep.on[0].dtype)
        ch = self.channels.index(channel)
        cs = self.channels.index(overlap) if overlap else None
        eye = jnp.eye(n_orb, dtype=jnp.result_type(kpts.dtype, jnp.complex64))

        def fn(p):
            readout, packs = eqx.combine(p, static)
            on = _readout(readout, F, atoms)
            off = OF._blocks_impl(packs, F, lay, b.rcut_bond, args, env)
            H = jax.vmap(lambda k: _bloch_impl(on, off, lists, k, ch, n_orb))(kpts)
            S = (jax.vmap(lambda k: _bloch_impl(on, off, lists, k, cs, n_orb))(kpts) + overlap_shift * eye if overlap
                 else jnp.broadcast_to(eye, H.shape))
            return jax.vmap(_eigvals_impl)(H, S)

        return jax.jit(fn), params

    def hk(self, prep, k=(0.0, 0.0, 0.0), channel="H"):
        """(n_orb, n_orb) complex Bloch matrix of `channel` at fractional k, on the device."""
        k = jnp.asarray(k, prep.on[0].dtype)
        return _bloch(prep.on, prep.off, prep.lists, k, self.channels.index(channel), prep.n_orb)

    def bands(self, prep, kpts, overlap="S", channel="H", chunk=64):
        """(n_k, n_orb) eigenvalues of H(k) c = e S(k) c at the fractional k-points `kpts`.
        `overlap=None`: an orthonormal basis (S = 1).  `channel`: the Hamiltonian channel
        ("H_up" or "H_dn" of a collinear-spin model).  The k-points are evaluated in batches of
        `chunk`: Bloch sums and eigensolves vmapped, nothing on the host."""
        kpts = jnp.asarray(np.atleast_2d(np.asarray(kpts, float)), prep.on[0].dtype)
        ch = self.channels.index(channel)
        Hk = jax.vmap(lambda k: _bloch_impl(prep.on, prep.off, prep.lists, k, ch, prep.n_orb))
        if overlap:
            cs = self.channels.index(overlap)
            Sk = jax.vmap(lambda k: _bloch_impl(prep.on, prep.off, prep.lists, k, cs, prep.n_orb))
        else:
            Sk = lambda ks: jnp.broadcast_to(jnp.eye(prep.n_orb, dtype=jnp.result_type(kpts.dtype, jnp.complex64)), (len(ks), prep.n_orb, prep.n_orb))
        return jnp.concatenate([_eigvals_k(Hk(kpts[i:i + chunk]), Sk(kpts[i:i + chunk])) for i in range(0, len(kpts), chunk)])
