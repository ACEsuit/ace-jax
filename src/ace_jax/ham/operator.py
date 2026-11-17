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
from . import bond as B
from . import offsite_fast as OF
from .offsite_fast import OffsitePredictor
from .onsite import OnsitePredictor, _features, _readout, site_graph


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
    phase = jnp.exp(2j * jnp.pi * (lists["R"] @ k)).astype(cdtype)             # (n, K)
    for key, blocks in off.items():
        w = phase * lists["valid"][key]
        M = _scatter(M, blocks[:, :, channel].astype(cdtype), lists["rows"], lists["cols"], w, n_orb)
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


def _device_impl(site, readout, packs, pos, cell, species, site_species, s_rij, s_idx, s_count, b_rij, b_idx, b_count, static):
    """Everything `prepare` does after the neighbour lists, on the device: site features, on-site
    blocks, the reverse slot and image of every bond, off-site blocks, and the index lists of the
    Bloch sum.  `species`: index into the orbital basis' species, per atom; `site_species`: index
    into the site basis' elements (its radials are tabulated in that order)."""
    args, counts, norbs, rcut_bond, pairs = static
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
    fmask = mask.astype(dtype)
    off = OF._blocks_impl(packs, F, b_rij, fmask, rev, rcut_bond, args)
    norb = jnp.asarray(norbs, jnp.int32)[species]
    offsets = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(norb)[:-1]])
    zj = species[b_idx]
    valid = {key: ((species[:, None] == ka) & (zj == kb) & mask).astype(jnp.float32)
             for key, (ka, kb) in zip(off, pairs)}
    lists = {"rows": jnp.broadcast_to(offsets[:, None], (n, K)), "cols": offsets[b_idx], "R": R.astype(dtype),
             "valid": valid, "on_rows": tuple(offsets[a] for a in atoms)}
    ok = jnp.all(jnp.where(mask, (b_idx.reshape(-1)[rev.reshape(-1)].reshape(n, K) == i), True))
    return on, off, lists, ok


_device = jax.jit(_device_impl, static_argnums=13)


# compiled, with the slot count static: an eager slice uploads its bounds as scalars
_trim = jax.jit(lambda rij, idx, kt: (rij[:, :kt], idx[:, :kt]), static_argnums=2)


class Prepared:
    """One structure, ready for `hk` and `bands`: the blocks on the device and the index lists
    that place them in the orbital matrix (`n_orb` orbitals; atom a starts at `offsets[a]`)."""

    def __init__(self, on, off, lists, n_orb, offsets, graph):
        self.on, self.off, self.lists, self.n_orb, self.offsets, self.graph = on, off, lists, n_orb, offsets, graph


class HamiltonianPredictor:
    """`onsite` and `offsite`: fitted models on the same site basis, orbital basis and channels."""

    def __init__(self, onsite, offsite):
        if onsite.channels != offsite.channels or onsite.orbitals != offsite.orbitals:
            raise ValueError("the on-site and off-site models have different channels or orbital bases")
        if onsite.basis is not offsite.bond.site:
            raise ValueError("the on-site and off-site models must share one site basis (the same object): "
                             "its features are evaluated once for both")
        self.onsite, self.offsite = OnsitePredictor(onsite), OffsitePredictor(offsite)
        self.channels, self.orbitals = onsite.channels, onsite.orbitals
        self._k = {}                                   # slots of the device neighbour lists, by cutoff

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
        pairs = tuple((orb.species.index(za), orb.species.index(zb)) for za, zb in self._pair_order())
        static = ((basis.n_bond, basis.n_cpl, basis.l_cpl, self.offsite.lmax), counts, norbs, basis.rcut_bond, pairs)
        pos_d, cell_d, sp_d, ss_d = (jax.device_put(np.asarray(positions, dtype)), jax.device_put(np.asarray(cell, dtype)),
                                     jax.device_put(species), jax.device_put(basis.site.species_index(numbers)))
        on, off, lists, ok = _device(basis.site, self.onsite.readout, self.offsite.packs, pos_d, cell_d, sp_d, ss_d,
                                     gs.rij, gs.idx, gs.count, gb.rij, gb.idx, gb.count, static)
        prep = Prepared(on, off, lists, int(sum(orb.n_orb(z) for z in numbers)), None, None)
        prep.reverse_ok = ok                              # device scalar: every bond found its reverse
        return prep

    def _pair_order(self):
        """The ordered species pairs in the order `offsite_fast._blocks` returns them."""
        out = []
        for za, zb in self.offsite.packs:
            out += [(za, zb)] if za == zb else [(za, zb), (zb, za)]
        return out

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
        count = np.bincount(g.i, minlength=n)
        K = max(int(count.max()) if len(g.i) else 1, 1)
        start = np.concatenate([[0], np.cumsum(count)])
        flat = g.i * K + (np.arange(len(g.i)) - start[g.i])
        rvec = np.zeros((n, K, 3)); rvec[..., 0] = basis.rcut_bond
        mask = np.zeros((n, K))
        nbr = np.zeros(n * K, np.int64)
        R = np.zeros((n * K, 3))
        rev = np.zeros(n * K, np.int32)
        rvec.reshape(-1, 3)[flat] = g.rij
        mask.reshape(-1)[flat] = 1.0
        nbr[flat], R[flat], rev[flat] = g.j, g.R, flat[g.reverse]
        off = self.offsite.device(F, rvec, mask, rev.reshape(n, K))
        offsets = np.concatenate([[0], np.cumsum([orb.n_orb(z) for z in numbers])])
        zj_slot = numbers[nbr].reshape(n, K)
        valid = {(za, zb): jnp.asarray(((numbers[:, None] == za) & (zj_slot == zb) & (mask > 0)).astype(np.float32))
                 for za, zb in off}
        lists = {"rows": jnp.asarray(np.repeat(offsets[:-1, None], K, 1), jnp.int32),
                 "cols": jnp.asarray(offsets[nbr].reshape(n, K), jnp.int32),
                 "R": jnp.asarray(R.reshape(n, K, 3), off[next(iter(off))].dtype if off else jnp.float32),
                 "valid": valid,
                 "on_rows": tuple(jnp.asarray(offsets[a], jnp.int32) for a in atoms)}
        return Prepared(on, off, lists, int(offsets[-1]), offsets, g)

    def hk(self, prep, k=(0.0, 0.0, 0.0), channel="H"):
        """(n_orb, n_orb) complex Bloch matrix of `channel` at fractional k, on the device."""
        k = jnp.asarray(k, prep.lists["R"].dtype)
        return _bloch(prep.on, prep.off, prep.lists, k, self.channels.index(channel), prep.n_orb)

    def bands(self, prep, kpts, overlap="S"):
        """(n_k, n_orb) eigenvalues of H(k) c = e S(k) c at the fractional k-points `kpts`.
        `overlap=None`: an orthonormal basis (S = 1)."""
        out = []
        for k in np.atleast_2d(np.asarray(kpts, float)):
            H = self.hk(prep, k, "H")
            S = self.hk(prep, k, overlap) if overlap else jnp.eye(prep.n_orb, dtype=H.dtype)
            out.append(_eigvals(H, S))
        return jnp.stack(out)
