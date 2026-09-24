import itertools, pathlib
import jax
import numpy as np
import pytest
from conftest import pace_fixture

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval.harmonics import real_spherical_harmonics
from ace_jax.eval.pace_build import build_basis, ylm_map

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"


def test_ylm_map_matches_pace_cpp():
    ref_p = FIX / "unit_ref.npz"
    pace_fixture(ref_p)
    ref = np.load(ref_p)
    U = ylm_map(6)
    YR = np.asarray(real_spherical_harmonics(jnp.asarray(ref["ydir"]), 6))
    YP = ref["ylm_re"] + 1j * ref["ylm_im"]
    for l in range(7):
        s = slice(l * l, (l + 1) ** 2)
        np.testing.assert_allclose(YR[:, s] @ U[l].T, YP[:, s], atol=1e-13)


def _toy_arrays(rng, NZ=2, K=3, nradmax=2, lmax=2):
    """Random functions with random (not necessarily invariant) ms combos."""
    funcs = [{"el": 0, "rank": 1, "mus": [1], "ns": [2], "ls": [0], "ms": np.array([[0]])}]
    for el in range(NZ):
        for rank in (2, 3):
            ls = list(rng.integers(0, lmax + 1, rank))
            ms = np.array([[rng.integers(-l, l + 1) for l in ls] for _ in range(4)])
            funcs.append({"el": el, "rank": rank, "mus": list(rng.integers(0, NZ, rank)),
                          "ns": list(rng.integers(1, nradmax + 1, rank)), "ls": ls, "ms": ms})
    n_terms = sum(len(f["ms"]) for f in funcs)
    return {"funcs": funcs, "nradbase": K, "nradmax": nradmax, "lmax": lmax,
            "Z": np.arange(NZ), "ndensity": 2,
            "ctilde_complex": rng.normal(size=(n_terms, 2))}


def test_real_basis_reproduces_complex_products():
    """sum_t Re(c_t prod A^PACE) == AA . ctilde_real, for random real A."""
    rng = np.random.default_rng(3)
    a = _toy_arrays(rng)
    b = build_basis(a)
    NZ, K, nm, lmax = 2, a["nradbase"], a["nradmax"], a["lmax"]
    ncol = K + nm * (lmax + 1)
    # random real "pooled" A per (mu, radial column, real Ylm column)
    Areal = rng.normal(size=(NZ, ncol, (lmax + 1) ** 2))
    U = ylm_map(lmax)

    def A_pace(mu, col, l, m):                     # complex A for PACE index m
        s = slice(l * l, (l + 1) ** 2)
        return U[l][m + l] @ Areal[mu, col, s]

    rho = np.zeros((NZ, 2))                        # [centre element, density]
    t = 0
    for f in a["funcs"]:
        for ms in f["ms"]:
            prod = 1.0 + 0j
            for k in range(f["rank"]):
                col = (f["ns"][k] - 1) if f["rank"] == 1 else K + (f["ns"][k] - 1) * (lmax + 1) + f["ls"][k]
                prod *= A_pace(f["mus"][k], col, f["ls"][k], ms[k])
            rho[f["el"]] += (prod * a["ctilde_complex"][t]).real
            t += 1

    Aflat = np.stack([Areal[mu][b["a_rad"], b["a_y"]] for mu in range(NZ)]).reshape(-1)
    AA = np.concatenate([np.prod(Aflat[s], axis=-1) for s in b["aa_specs"]])
    ct = np.zeros((b["n_aa"] * NZ, 2))
    np.add.at(ct, b["T_rows"], b["T_vals"][:, None] * a["ctilde_complex"][b["T_cols"]])
    ct = ct.reshape(b["n_aa"], NZ, 2)
    np.testing.assert_allclose(np.einsum("a,aep->ep", AA, ct), rho, rtol=1e-12, atol=1e-12)


def test_aa_specs_sorted_and_unique():
    b = build_basis(_toy_arrays(np.random.default_rng(5)))
    keys = [tuple(r) for s in b["aa_specs"] for r in np.asarray(s)]
    assert len(keys) == len(set(keys))
    assert all(list(k) == sorted(k) for k in keys)
