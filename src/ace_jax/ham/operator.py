"""From predicted blocks to H(k), S(k) and eigenvalues, on the device.

`HamiltonianPredictor` runs the on-site and off-site models on one set of site features and
keeps the blocks where they were computed.  `hk` sums them into the Bloch matrices
M(k) = sum_R exp(2 pi i k.R) M(R) (k in fractional coordinates of the reciprocal cell) and
`bands` solves H(k) c = e S(k) c through a Cholesky factor of S(k).  Nothing the size of the
blocks crosses to the host: the lists go up (a few MB), the eigenvalues come down."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..eval.nlist import dense_graph, have_matscipy_neighbours, sparse_graph
from ..operators import bond as B
from ..operators import offsite_fast as OF
from ..operators.offsite_fast import OffsitePredictor
from ..operators.onsite import OnsitePredictor, _features, _readout, site_graph
from ..operators.precision import highest


def _scatter(M, blocks, rows, cols, weight, n_orb):
    """Add `blocks` (..., n_a, n_b), each times its `weight` (...,), into the flat (n_orb^2,)
    matrix `M` at row offsets `rows` and column offsets `cols` (...,)."""
    na, nb = blocks.shape[-2:]
    idx = ((rows[..., None, None] + jnp.arange(na)[:, None]) * n_orb + cols[..., None, None] + jnp.arange(nb))
    return M.at[idx.reshape(-1)].add((blocks * weight[..., None, None]).reshape(-1))


@highest
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


@highest
def _eigvals_impl(H, S):
    """Eigenvalues of H c = e S c, both Hermitian and S positive definite."""
    H, S = 0.5 * (H + H.conj().T), 0.5 * (S + S.conj().T)
    L = jnp.linalg.cholesky(S)
    X = jax.scipy.linalg.solve_triangular(L, H, lower=True)
    Ht = jax.scipy.linalg.solve_triangular(L, X.conj().T, lower=True).conj().T
    return jnp.linalg.eigvalsh(0.5 * (Ht + Ht.conj().T))


_eigvals = jax.jit(_eigvals_impl)
_eigvals_k = jax.jit(jax.vmap(_eigvals_impl))             # a batch of k-points in one call


@highest
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


@highest
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


class BandEnergy(NamedTuple):
    """energy: E_band = sum_k w_k sum_n f e; gradient: d free_energy / d positions (n_atoms, 3), the
    force is its negative; stress: (1 / V) d free_energy / d strain (3, 3), symmetric, at fixed
    fractional k-points (energy per volume: eV / angstrom^3 for eV and angstrom); eigenvalues
    (n_k, n_orb); free_energy: E_band - T S of the smearing (E_band for fixed occupations), the
    quantity the gradient and stress are derivatives of; fermi_level (nan for fixed occupations)."""
    energy: jax.Array
    gradient: jax.Array
    stress: jax.Array
    eigenvalues: jax.Array
    free_energy: jax.Array = None
    fermi_level: jax.Array = None


SMEARINGS = ("fermi-dirac", "gaussian")


def smeared_occupations(e, wk, nelec, width, kind="fermi-dirac", degeneracy=2.0):
    """(f, mu, -T S): occupations f (n_k, n) in [0, degeneracy] of the bands e with k-point weights
    wk that hold `nelec` electrons, the chemical potential (bisection), and the entropy term of
    the free energy.  fermi-dirac: f = g / (1 + exp((e - mu) / width)), width = k_B T.
    gaussian: f = g erfc((e - mu) / width) / 2 (FHI-aims' `occupation_type gaussian`)."""
    def occ(mu):
        x = (e - mu) / width
        return degeneracy * (jax.nn.sigmoid(-x) if kind == "fermi-dirac" else 0.5 * jax.scipy.special.erfc(x))

    def step(_, b):
        lo, hi = b
        mid = 0.5 * (lo + hi)
        low = jnp.sum(wk[:, None] * occ(mid)) < nelec
        return jnp.where(low, mid, lo), jnp.where(low, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 100, step, (jnp.min(e) - 20.0 * width, jnp.max(e) + 20.0 * width))
    mu = 0.5 * (lo + hi)
    f = occ(mu)
    if kind == "fermi-dirac":
        x = jnp.clip(f / degeneracy, 1e-300, 1.0)
        y = jnp.clip(1.0 - f / degeneracy, 1e-300, 1.0)
        mts = degeneracy * width * jnp.sum(wk[:, None] * (x * jnp.log(x) + y * jnp.log(y)))
    else:
        mts = -degeneracy * width / (2.0 * jnp.sqrt(jnp.pi)) * jnp.sum(wk[:, None] * jnp.exp(-((e - mu) / width) ** 2))
    return f, mu, mts


def _occupation_fn(kind, degeneracy, params):
    """occ(e, wk) -> (f, mu, -T S) for `kind` "fixed" (params: (f,)) or a smearing (params: (nelec,
    width[, n_core])).  n_core: the lowest bands left out of the band energy (their occupations are
    returned as zero, after the chemical potential is found with every electron)."""
    if kind == "fixed":
        return lambda e, wk: (jnp.broadcast_to(params[0], e.shape), jnp.asarray(jnp.nan, e.dtype), jnp.zeros((), e.dtype))

    def occ(e, wk):
        f, mu, mts = smeared_occupations(e, wk, params[0], params[1], kind, degeneracy)
        if len(params) > 2:
            f = jnp.where(jnp.arange(e.shape[-1])[None, :] < params[2], 0.0, f)
        return f, mu, mts
    return occ


@highest
def _geig(H, S):
    """Eigenvalues and S-orthonormal eigenvectors of H C = S C e."""
    L = jnp.linalg.cholesky(S)
    A = jax.scipy.linalg.solve_triangular(L, H, lower=True)
    A = jax.scipy.linalg.solve_triangular(L, A.conj().T, lower=True).conj().T
    e, U = jnp.linalg.eigh(0.5 * (A + A.conj().T))
    return e, jax.scipy.linalg.solve_triangular(L.conj().T, U, lower=False)


@highest
def _band_energy(hs, pos, occ, wk, volume):
    """`BandEnergy` of hs(positions, strain) -> (H(k), S(k)) by the trace form: the eigenproblem is
    solved once and not differentiated; one reverse pass gives the position gradient and the strain
    derivative together.  The strain scales every edge vector, v -> v (1 + strain)^T, which is what
    straining positions and cell together does; the Bloch phases depend on the cells only."""
    def trace(x):
        H, S = hs(*x)
        e, C = jax.vmap(_geig)(jax.lax.stop_gradient(H), jax.lax.stop_gradient(S))
        f, mu, mts = jax.tree.map(jax.lax.stop_gradient, occ(e, wk))
        f = f * wk[:, None]
        rho = jnp.einsum("kin,kn,kjn->kij", C, f, C.conj())
        W = jnp.einsum("kin,kn,kjn->kij", C, f * e, C.conj())
        # tr[rho H - W S]: zero in value; its derivative at fixed rho, W is dE_band at fixed
        # occupations, which with smeared occupations is the derivative of the free energy
        return jnp.real(jnp.sum(H * jnp.swapaxes(rho, 1, 2)) - jnp.sum(S * jnp.swapaxes(W, 1, 2))), (jnp.sum(f * e), e, mu, mts)

    (gpos, gstrain), (E, e, mu, mts) = jax.grad(trace, has_aux=True)((pos, jnp.zeros((3, 3), pos.dtype)))
    return BandEnergy(E, gpos, 0.5 * (gstrain + gstrain.T) / volume, e, E + mts, mu)


@highest
def _device_energy_impl(site, readout, packs, pos, cell, species, site_species, s_rij, s_idx, s_count, b_rij, b_idx, b_count,
                        env, e_idx, kpts, occ, wk, static):  # occ: the parameters of `_occupation_fn`
    """`_band_energy` from device neighbour lists: the list vectors are rebuilt from the positions
    (v = x_j - x_i + shift, the shift read off the list and held fixed), so the derivative with
    respect to the positions and the strain flows through them."""
    dev_static, n_orb, ch, cs, kind, degeneracy = static

    def vectors(p, strain, r0, idx):
        d = p[idx] - p[:, None, :]
        v = d + jax.lax.stop_gradient(r0 - d)
        return v + v @ strain.T

    def hs(p, strain):
        e = None if env is None else (vectors(p, strain, env[0], e_idx), env[1], env[2])
        on, off, lists, _ok = _device_impl(site, readout, packs, jax.lax.stop_gradient(p), cell, species, site_species,
                                           vectors(p, strain, s_rij, s_idx), s_idx, s_count,
                                           vectors(p, strain, b_rij, b_idx), b_idx, b_count, e, dev_static)
        H = jax.vmap(lambda k: _bloch_impl(on, off, lists, k, ch, n_orb))(kpts)
        S = (jax.vmap(lambda k: _bloch_impl(on, off, lists, k, cs, n_orb))(kpts) if cs is not None
             else jnp.broadcast_to(jnp.eye(n_orb, dtype=H.dtype), H.shape))
        herm = lambda M: 0.5 * (M + jnp.swapaxes(M.conj(), 1, 2))
        return herm(H), herm(S)

    return _band_energy(hs, pos, _occupation_fn(kind, degeneracy, occ), wk, jnp.abs(jnp.linalg.det(cell)))


_device_energy = jax.jit(_device_energy_impl, static_argnums=18)


@highest
def _host_hs(site, readout, packs, pos, strain, G, kpts, static, overlap_shift=0.0):
    """(H(k), S(k)), Hermitian, from host neighbour lists `G` (`HamiltonianPredictor._host_lists`): every
    edge vector is rebuilt from the positions, v = x_j - x_i + shift, and scaled by the strain."""
    args, n_orb, ch, cs, rcut_bond = static
    st = (lambda v: v) if strain is None else (lambda v: v + v @ strain.T)
    n = pos.shape[0]
    F = site.features(st(pos[G["s_j"]] - pos[G["s_i"]] + G["s_shift"]), G["sp"][G["s_i"]], G["sp"][G["s_j"]], G["s_i"], n, G["s_mask"])
    on = _readout(readout, F, G["atoms"])
    pad = jnp.asarray([rcut_bond, 0.0, 0.0], pos.dtype)
    lay = {key: (rows, jnp.where(live[..., None], st(pos[bj] - pos[bi] + shift), pad), mask, rev)
           for key, (rows, bi, bj, shift, live, mask, rev) in G["lay"].items()}
    env = None
    if G["env"] is not None:
        e_rij, e_idx, e_shift, e_live, e_count, e_z = G["env"]
        env = (jnp.where(e_live[..., None], st(pos[e_idx] - pos[:, None, :] + e_shift), e_rij), e_count, e_z)
    off = OF._blocks_impl(packs, F, lay, rcut_bond, args, env)
    H = jax.vmap(lambda k: _bloch_impl(on, off, G["bloch"], k, ch, n_orb))(kpts)
    eye = jnp.eye(n_orb, dtype=H.dtype)
    S = (jax.vmap(lambda k: _bloch_impl(on, off, G["bloch"], k, cs, n_orb))(kpts) + overlap_shift * eye if cs is not None
         else jnp.broadcast_to(eye, H.shape))
    herm = lambda M: 0.5 * (M + jnp.swapaxes(M.conj(), 1, 2))
    return herm(H), herm(S)


@highest
def _host_energy_impl(site, readout, packs, pos, cell, G, kpts, occ, wk, static):
    hs_static, kind, degeneracy = static
    return _band_energy(lambda p, strain: _host_hs(site, readout, packs, p, strain, G, kpts, hs_static), pos,
                        _occupation_fn(kind, degeneracy, occ), wk, jnp.abs(jnp.linalg.det(cell)))


_host_energy = jax.jit(_host_energy_impl, static_argnums=9)        # compiled once per set of list shapes


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

    @classmethod
    def from_file(cls, path, threads=1):
        """The predictor of a model file (`operators.io`) that holds an on-site and an off-site model."""
        from ..operators import io
        f = io.load(path)
        if f.onsite is None or f.offsite is None:
            raise ValueError(f"{path}: a Hamiltonian needs the on-site and the off-site model; the file has "
                             f"{'no on-site' if f.onsite is None else 'no off-site'} model")
        return cls(f.onsite, f.offsite, threads=threads)

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
        a = self._device_inputs(positions, cell, numbers, pbc)
        on, off, lists, ok = _device(*a["args"], a["env"], a["static"])
        prep = Prepared(on, off, lists, a["n_orb"], None, None)
        prep.reverse_ok = ok                              # device scalar: every bond found its reverse
        return prep

    def _device_inputs(self, positions, cell, numbers, pbc):
        """The device neighbour lists and the arguments of `_device` for a structure."""
        orb, basis = self.orbitals, self.offsite.model.bond
        numbers = np.asarray(numbers)
        if not all(int(z) in orb.species for z in numbers):
            raise KeyError("the structure has a species without an orbital basis")
        species = np.asarray([orb.species.index(int(z)) for z in numbers], np.int32)
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
        env = e_idx = None
        if basis.n_env:
            ge = self._dense(positions, cell, pbc, basis.r_env)
            env, e_idx = (ge.rij, ge.count, _take_species(ss_d, ge.idx)), ge.idx
        return dict(args=(basis.site, self.onsite.readout, self.offsite.packs, pos_d, cell_d, sp_d, ss_d,
                          gs.rij, gs.idx, gs.count, gb.rij, gb.idx, gb.count),
                    env=env, e_idx=e_idx, static=static, n_orb=int(sum(orb.n_orb(z) for z in numbers)), dtype=dtype)

    def band_energy_device(self, positions, cell, numbers, kpts, occupations, weights=None, overlap="S", channel="H", pbc=True):
        """`BandEnergy` (energy, position gradient, stress, eigenvalues) with the neighbour lists
        built on the GPU and everything derived from them there, as `prepare_device`: for
        structures that change every call (MD).  The uploads are the positions, the cell, the
        species, the k-points and the occupations; the lists are rebuilt on each call and the
        program is compiled once per set of list shapes.  occupations: (n_k, n_orb) or (n_orb,)
        fixed occupations (a function of the eigenvalues is not taken here)."""
        if jax.default_backend() != "gpu" or not have_matscipy_neighbours():
            raise RuntimeError("band_energy_device needs JAX on a GPU and matscipy-neighbours built with CUDA")
        a = self._device_inputs(positions, cell, numbers, pbc)
        kpts = np.atleast_2d(np.asarray(kpts, a["dtype"]))
        wk = np.full(len(kpts), 1.0 / len(kpts), a["dtype"]) if weights is None else np.asarray(weights, a["dtype"])
        static = (a["static"], a["n_orb"], self.channels.index(channel), self.channels.index(overlap) if overlap else None,
                  "fixed", 2.0)
        return _device_energy(*a["args"], a["env"], a["e_idx"], jax.device_put(kpts),
                              (jax.device_put(np.asarray(occupations, a["dtype"])),), jax.device_put(wk), static)

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

    def _hs_fn(self, positions, cell, numbers, kpts, overlap="S", channel="H", overlap_shift=0.0, pbc=True):
        """(hs, positions): hs(positions) -> (H(k), S(k)) at the fractional k-points as a traceable
        function of the atomic positions, through the edge vectors of the site features, the bond
        and environment geometry and the Bloch sums.

        The neighbour topology (which pairs are site neighbours, bonds and environment atoms, and
        in which cells) is that of the given positions and is held fixed: every feature vanishes
        smoothly at its cutoff, so the derivative at these positions is exact, and fn is valid
        while no pair crosses a cutoff.  The Bloch phases use the cells R only (the lattice
        gauge); eigenvalues do not depend on the gauge.  overlap, overlap_shift: as `spectrum_fn`."""
        G, n_orb = self._host_lists(positions, cell, numbers, pbc)
        static = (self.offsite.static(), n_orb, self.channels.index(channel), self.channels.index(overlap) if overlap else None,
                  self.offsite.model.bond.rcut_bond)
        site, readout, packs = self.offsite.model.bond.site, self.onsite.readout, self.offsite.packs
        kpts = jnp.asarray(np.atleast_2d(np.asarray(kpts, float)))
        hs = lambda pos, strain=None: _host_hs(site, readout, packs, pos, strain, G, kpts, static, overlap_shift)
        return hs, jnp.asarray(np.asarray(positions, float))

    def _host_lists(self, positions, cell, numbers, pbc=True, edge_step=1, k_step=1):
        """(G, n_orb): the host neighbour lists of a structure as device arrays, in the form
        `_host_hs` rebuilds the edge vectors from.  edge_step, k_step: pad the site edge list and
        the bond / environment slots up to multiples of these, so that structures whose neighbour
        counts differ a little share one compiled program."""
        orb, basis = self.orbitals, self.offsite.model.bond
        numbers = np.asarray(numbers)
        pos0, cell = np.asarray(positions, float), np.asarray(cell, float)
        n = len(numbers)
        for z in set(int(z) for z in numbers):
            orb.ls(z)
        gs = sparse_graph(pos0, cell, pbc, basis.site.rcut)
        E = max(-(-len(gs.senders) // edge_step) * edge_step, 1)
        npad = E - len(gs.senders)
        s_i, s_j = (np.concatenate([a, np.zeros(npad, a.dtype)]) for a in (gs.senders, gs.receivers))
        s_shift = np.concatenate([gs.shifts, np.tile([basis.site.rcut, 0.0, 0.0], (npad, 1))])       # padding: at the cutoff
        s_mask = np.arange(E) < len(gs.senders)
        atoms = tuple(np.flatnonzero(numbers == z).astype(np.int32) for z in orb.species)
        g = B.bond_graph(basis, pos0, cell, numbers, pbc)
        pl = OF.PairLayouts(g, numbers, orb.species, basis.rcut_bond, k_step=k_step)
        offsets = np.concatenate([[0], np.cumsum([orb.n_orb(z) for z in numbers])])
        lay0, bloch_off = {}, {}
        for key, (rows, _rvec, mask, rev) in pl.lay.items():
            b = pl.bond[key]
            live, e = b >= 0, np.maximum(b, 0)
            lay0[key] = (jnp.asarray(rows), jnp.asarray(g.i[e]), jnp.asarray(g.j[e]), jnp.asarray(g.R[e] @ cell),
                         jnp.asarray(live), jnp.asarray(mask), jnp.asarray(rev))
            bloch_off[key] = (jnp.asarray(offsets[g.i[e]], jnp.int32), jnp.asarray(offsets[g.j[e]], jnp.int32),
                              jnp.asarray(np.where(live[..., None], g.R[e], 0), float), jnp.asarray(live.astype(np.float32)))
        env0 = None
        if basis.n_env:
            ge = sparse_graph(pos0, cell, pbc, basis.r_env)
            K = max(int(np.bincount(ge.senders, minlength=n).max()) if len(ge.rij) else 1, 1)
            K = -(-K // k_step) * k_step
            d = dense_graph(pos0, cell, pbc, basis.r_env, K)
            e_rij, e_idx, e_count = np.asarray(d.rij), np.asarray(d.idx), np.asarray(d.count)
            e_live = np.arange(K)[None, :] < e_count[:, None]
            e_shift = np.where(e_live[..., None], e_rij - (pos0[e_idx] - pos0[:, None, :]), 0.0)
            env0 = (jnp.asarray(e_rij), jnp.asarray(e_idx), jnp.asarray(e_shift), jnp.asarray(e_live), jnp.asarray(e_count),
                    jnp.asarray(basis.site.species_index(numbers)[e_idx]))
        G = dict(sp=jnp.asarray(basis.site.species_index(numbers)), s_i=jnp.asarray(s_i), s_j=jnp.asarray(s_j),
                 s_shift=jnp.asarray(s_shift), s_mask=jnp.asarray(s_mask), atoms=tuple(jnp.asarray(a) for a in atoms), lay=lay0, env=env0,
                 bloch={"off": bloch_off, "on_rows": tuple(jnp.asarray(offsets[a], jnp.int32) for a in atoms)})
        return G, int(offsets[-1])

    def band_energy(self, positions, cell, numbers, kpts, nelectrons, width=0.1, smearing="fermi-dirac", weights=None,
                    overlap="S", channel="H", degeneracy=2.0, device=False, pbc=True, core_bands=0):
        """`BandEnergy` of a structure with smeared occupations: band energy, free energy, the
        gradient and stress of the free energy (trace form), eigenvalues and Fermi level, in one
        compiled call.  nelectrons: the electrons the bands hold (of the orbitals the model has:
        all of them for an all-electron basis).  smearing: "fermi-dirac" (width = k_B T) or
        "gaussian"; degeneracy: 2 without spin.  The program is compiled once per set of list
        shapes (the lists are padded in steps), so a trajectory re-uses it.  device=True: the
        neighbour lists on the GPU (`band_energy_device`'s path); otherwise built on the host.
        core_bands: the lowest bands of the cell left out of the band energy and its derivatives
        (a valence band energy from an all-electron basis); the Fermi level still counts them."""
        if smearing not in SMEARINGS:
            raise ValueError(f"smearing must be one of {SMEARINGS}, got {smearing!r}")
        kpts = np.atleast_2d(np.asarray(kpts, float))
        wk = np.full(len(kpts), 1.0 / len(kpts)) if weights is None else np.asarray(weights, float)
        occ = (np.float64(nelectrons), np.float64(width), np.int32(core_bands))            # host scalars: uploaded explicitly below
        ch, cs = self.channels.index(channel), self.channels.index(overlap) if overlap else None
        if device:
            if jax.default_backend() != "gpu" or not have_matscipy_neighbours():
                raise RuntimeError("device=True needs JAX on a GPU and matscipy-neighbours built with CUDA")
            a = self._device_inputs(positions, cell, numbers, pbc)
            dt = a["dtype"]
            return _device_energy(*a["args"], a["env"], a["e_idx"], jax.device_put(kpts.astype(dt)),
                                  (jax.device_put(occ[0].astype(dt)), jax.device_put(occ[1].astype(dt)), jax.device_put(occ[2])),
                                  jax.device_put(wk.astype(dt)),
                                  (a["static"], a["n_orb"], ch, cs, smearing, float(degeneracy)))
        G, n_orb = self._host_lists(positions, cell, numbers, pbc, edge_step=1024, k_step=8)
        b = self.offsite.model.bond
        static = ((self.offsite.static(), n_orb, ch, cs, b.rcut_bond), smearing, float(degeneracy))
        return _host_energy(b.site, self.onsite.readout, self.offsite.packs, jnp.asarray(np.asarray(positions, float)),
                            jnp.asarray(np.asarray(cell, float)), G, jnp.asarray(kpts), tuple(jnp.asarray(x) for x in occ),
                            jnp.asarray(wk), static)

    def bands_fn(self, positions, cell, numbers, kpts, overlap="S", channel="H", overlap_shift=0.0, pbc=True):
        """(fn, positions): fn(positions) -> (n_k, n_orb) eigenvalues, differentiable through the
        eigensolver (`_hs_fn` says what is held fixed).  For the force of a band energy use
        `band_energy_fn`, which does not differentiate the eigensolver."""
        hs, pos0 = self._hs_fn(positions, cell, numbers, kpts, overlap, channel, overlap_shift, pbc)
        return jax.jit(lambda pos: jax.vmap(_eigvals_impl)(*hs(pos))), pos0

    def band_energy_fn(self, positions, cell, numbers, kpts, occupations, weights=None, overlap="S", channel="H",
                       overlap_shift=0.0, pbc=True):
        """(fn, positions): fn(positions) -> `BandEnergy` (energy, gradient, stress, eigenvalues),
        with E_band = sum_k w_k sum_n f_kn e_kn; the force is minus the gradient, and the stress is
        (1 / V) dE / d strain from the same reverse pass (every edge vector scaled by the strain).

        The gradient is the trace form (Hellmann-Feynman for a non-orthogonal basis),
            dE = sum_k w_k Re tr[rho_k dH_k - W_k dS_k],
            rho_k = C f C^+,  W_k = C (f e) C^+,  H C = S C e,
        so the eigenproblem is solved once and never differentiated: one reverse pass through the
        Bloch sums and the block predictors, no 1 / (e_n - e_m) terms, finite at degeneracies.
        occupations: (n_k, n_orb) or (n_orb,) fixed f, or a function of the eigenvalues (evaluated
        at the current eigenvalues and held fixed in the derivative: the force at fixed
        occupations).  weights: k-point weights (default 1 / n_k).  The rest as `bands_fn`."""
        hs, pos0 = self._hs_fn(positions, cell, numbers, kpts, overlap, channel, overlap_shift, pbc)
        nk = len(np.atleast_2d(np.asarray(kpts, float)))
        wk = jnp.full(nk, 1.0 / nk) if weights is None else jnp.asarray(weights)
        if callable(occupations):
            occ = lambda e, w: (occupations(e), jnp.asarray(jnp.nan, e.dtype), jnp.zeros((), e.dtype))
        else:
            occ = _occupation_fn("fixed", 2.0, (jnp.asarray(occupations),))

        volume = float(abs(np.linalg.det(np.asarray(cell, float))))
        return jax.jit(lambda pos: _band_energy(hs, pos, occ, wk, volume)), pos0

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
