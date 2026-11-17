"""From predicted blocks to H(k), S(k) and eigenvalues, on the device.

`HamiltonianPredictor` runs the on-site and off-site models on one set of site features and
keeps the blocks where they were computed.  `hk` sums them into the Bloch matrices
M(k) = sum_R exp(2 pi i k.R) M(R) (k in fractional coordinates of the reciprocal cell) and
`bands` solves H(k) c = e S(k) c through a Cholesky factor of S(k).  Nothing the size of the
blocks crosses to the host: the lists go up (a few MB), the eigenvalues come down."""
import jax
import jax.numpy as jnp
import numpy as np

from . import bond as B
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
