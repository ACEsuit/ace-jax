"""The design rows of the off-site fit as one compiled program per species pair: `bond.features`
and `bond.design` in JAX, for a fixed-size chunk of bonds, on whatever device JAX runs on.

`bond.py` is the reference (NumPy, one symmetry at a time); this module returns the same
matrices (tests/test_operators_design.py holds them to 1e-12) for all the designs of a species
pair at once, so that a fit never leaves the device between the site features and the QR
merges.  Padding bonds carry mask 0 and give zero rows."""
import jax
import jax.numpy as jnp
import numpy as np

from ..basis import cg
from ..eval.harmonics import real_spherical_harmonics
from . import bond as B
from .offsite_fast import _radial
from .precision import highest


@highest
def _env_half(basis, emb, e_rij, e_count, e_z, atoms, r):
    """`bond._env_side`: (e, channels n_env, (l_env+1)^2) for the bonds' first atoms `atoms`."""
    d = e_rij[atoms]                                                   # (e, K, 3)
    K = d.shape[1]
    dist = jnp.sqrt(jnp.sum(d * d, axis=-1))
    partner = jnp.sum((d - r[:, None, :]) ** 2, axis=-1) < 1e-12
    live = (jnp.arange(K)[None, :] < e_count[atoms][:, None]) & ~partner
    w = jnp.where(live & (dist < basis.r_env), (1 - jnp.clip(dist / basis.r_env, 0, 1) ** 2) ** 2, 0.0)
    x = d - 0.5 * r[:, None, :]
    t, Y = B.env_point(jnp, x, basis.r_env + 0.5 * basis.rcut_bond, basis.l_env, basis.env_core)
    R = _radial(jnp.clip(t, 0.0, 1.0), basis.n_env)
    nlm = (basis.l_env + 1) ** 2
    return jnp.einsum("ek,ekc,ekn,eky->ecny", w, emb[e_z[atoms]], R, Y).reshape(r.shape[0], -1, nlm)


def bond_designs(basis, keys):
    """A compiled function (F, r, i, j, mask, env) -> tuple of design matrices, one per (L, p, same)
    of `keys`, each (n_bonds * (2L+1), n_columns): rows bond-major, as `fit_offsite` stacks
    `bond.design`.  F: the site features {(lam, pi): (n_atoms, n_B, 2 lam + 1)}; r (n_bonds, 3), i, j
    (n_bonds,): the bonds; mask (n_bonds,): 0 on padding; env: `bond.env_lists` or None."""
    keys = tuple(keys)
    lps = sorted({(L, p) for L, p, _ in keys})
    lmax = max([L for L, _ in lps] + [basis.l_cpl, basis.l_ebond, 1])
    emb = B.env_embedding(basis) if basis.n_env else None
    n_ch = emb.shape[1] if basis.n_env else 0

    @highest
    def impl(F, r, i, j, mask, env):
        dist = jnp.sqrt(jnp.sum(r * r, axis=-1))
        x = jnp.clip(dist / basis.rcut_bond, 0.0, 1.0)
        Y = real_spherical_harmonics(r, lmax)
        y = lambda l: Y[:, l * l:(l + 1) ** 2]
        phi0, phi1 = _radial(x, basis.n_bond), _radial(x, basis.n_cpl)
        e = r.shape[0]
        if basis.n_env:
            e_rij, e_count, e_z = env
            em = jnp.asarray(emb, r.dtype)
            Aenv = _env_half(basis, em, e_rij, e_count, e_z, i, r) + _env_half(basis, em, e_rij, e_count, e_z, j, -r)
            phi2 = _radial(x, basis.n_ebond)
        feats = {}
        for L, p in lps:
            two, cpl = B.layout(basis, L, p)
            k = 2 * L + 1
            empty = jnp.zeros((e, 0, k), r.dtype)
            bond = phi0[:, :, None] * y(L)[:, None, :] if two else empty
            gi, gj, sign = [], [], []
            for l, lam, pi, nB in cpl:
                W = jnp.einsum("em,mnM->enM", y(l), jnp.asarray(cg.real_cg(l, lam, L), r.dtype))
                for site, acc in ((i, gi), (j, gj)):
                    g = jnp.einsum("ebn,enM->ebM", F[(lam, pi)][site], W)
                    acc.append((phi1[:, :, None, None] * g[:, None]).reshape(e, -1, k))
                sign += [(-1) ** l] * (basis.n_cpl * nB)
            ge, esign = [], []
            for l, lam in B.env_layout(basis, L, p):
                W = jnp.einsum("em,mnM->enM", y(l), jnp.asarray(cg.real_cg(l, lam, L), r.dtype))
                a = jnp.einsum("ebn,enM->ebM", Aenv[:, :, lam * lam:(lam + 1) ** 2], W)
                ge.append((phi2[:, :, None, None] * a[:, None]).reshape(e, -1, k))
                esign += [(-1) ** l] * (basis.n_ebond * n_ch * basis.n_env)
            feats[(L, p)] = (bond, jnp.concatenate(gi, axis=1) if gi else empty, jnp.concatenate(gj, axis=1) if gj else empty,
                             np.asarray(sign, float), jnp.concatenate(ge, axis=1) if ge else empty, np.asarray(esign, float))
        out = []
        for L, p, same in keys:
            bond, fi, fj, sign, fe, esign = feats[(L, p)]
            if not same:
                A = jnp.concatenate([bond, fi, fj, fe], axis=1)
            else:
                s = jnp.asarray(((-1) ** L) * sign, r.dtype)
                keep = np.flatnonzero(esign == (-1) ** L)
                A = jnp.concatenate([bond, 0.5 * (fi + s[None, :, None] * fj), fe[:, keep]], axis=1)
            out.append((jnp.swapaxes(A, 1, 2) * mask[:, None, None]).reshape(e * (2 * L + 1), -1))
        return tuple(out)

    return jax.jit(impl)


@jax.jit
@highest
def qr_merge(R, A, Y):
    """The triangular factor of [R; A | Y]: R (K + k, K + k) carries the right-hand sides as its
    last k columns (c = R[:K, K:]), so only R is formed."""
    return jnp.linalg.qr(jnp.concatenate([R, jnp.concatenate([A, Y], axis=1)], axis=0), mode="r")
