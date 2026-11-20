"""Profile of the total energy E_band + E_eff on a GPU: where the time goes and what crosses
between host and device.

E_band: `HamiltonianPredictor.band_energy(device=True)` (neighbour lists, site features, blocks,
Bloch sums, eigensolver, reverse pass for forces and stress).  E_eff: an ace-jax potential through
`ACECalculator` (the CUDA FFI kernel with ffi=True).  Model: H and S of core-projected aluminium
(9 orbitals per atom) fitted to two structures; the potential is the residual fit's.
    uv run python bench/ham/total_profile.py <projected npz dir> <potential model.npz> [n=1,2] [nk=2] [x64=1]
n: supercell multiples of the 64-atom cell along one axis (64, 128 atoms...)."""
import os
import sys
import time

import jax

X64 = dict(a.split("=") for a in sys.argv[3:]).get("x64", "1") == "1"
jax.config.update("jax_enable_x64", X64)

import jax.numpy as jnp
import numpy as np
from ase import Atoms

import ace_jax as aj
from ace_jax.ham import cache
from ace_jax.ham import operator as O
from ace_jax.operators import bond as B
from ace_jax.operators.features import build_site_basis
from ace_jax.operators.fit import fit_offsite, fit_onsite


def best(f, reps=5):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        out = f()
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t)
    return min(ts), out


def main(npz, potential, n=(1, 2), nk=2):
    jax.config.update("jax_enable_x64", True)                 # the fit needs float64
    train = [cache.load(os.path.join(npz, f))[0] for f in ("fcc_000.npz", "bcc_000.npz")]
    site = build_site_basis([13], order=1, totaldegree=40, lmax_feat=4, rcut=7.0, wL=0.5, nmax_by_order=(12,), lmax_by_order=(4,)).with_table()
    rmax = float(np.ceil(10 * max(np.linalg.norm(train[0].bond(k)) for k in train[0].blocks["S"])) / 10)
    basis = B.BondBasis(site=site, rcut_bond=rmax, n_bond=12, n_cpl=2, l_cpl=2, r_env=5.0, n_env=6, l_env=4, n_ebond=4, l_ebond=4)
    on, _ = fit_onsite(site, train)
    off, _ = fit_offsite(basis, train)
    jax.config.update("jax_enable_x64", X64)
    if not X64:
        import dataclasses
        cast = lambda t: jax.tree.map(lambda x: x.astype(jnp.float32) if hasattr(x, "dtype") and x.dtype == jnp.float64 else x, t)
        off = cast(off)
        on = dataclasses.replace(cast(on), basis=off.bond.site)              # one site basis object for both
    hp = O.HamiltonianPredictor(on, off)
    d = train[0]
    ks = np.array([[a, b, c] for a in range(nk) for b in range(nk) for c in range(nk)]) / nk
    dev = jax.devices()[0]
    print(f"{dev.platform} ({dev.device_kind}), {'float64' if X64 else 'float32'}; H model: 9 orbitals per atom, bond cutoff {rmax}, site (2, 2) + env (5, 6, 4, 4, 4)")
    for m in n:
        one(hp, d, ks, potential, m)


def one(hp, d, ks, potential, m):
    a = Atoms(numbers=d.numbers, positions=d.positions, cell=d.cell, pbc=True) * (m, 1, 1)
    kk = ks if m == 1 else ks[: max(1, len(ks) // m)]
    pos, cell, numbers = a.positions, np.asarray(a.cell), a.numbers
    N, nel = len(a), 3.0 * len(a)
    full = lambda: hp.band_energy(pos, cell, numbers, kk, nel, width=0.1, smearing="gaussian", device=True)
    t_full, out = best(full)
    # transfers: nothing implicit may go up; what comes down is logged by JAX to stderr
    with jax.transfer_guard_host_to_device("disallow"):
        full()
    down = "none"
    try:
        with jax.transfer_guard_device_to_host("disallow"):
            jax.block_until_ready(full())
    except Exception as e:                                      # the first implicit download, and where
        import traceback
        where = [f for f in traceback.extract_tb(e.__traceback__) if "ace_jax" in f.filename]
        down = f"{where[-1].name}, {os.path.basename(where[-1].filename)}:{where[-1].lineno}: {where[-1].line}" if where else str(e)[:80]
    print(f"\nimplicit host-to-device transfers in a warm call: none (guard passed); first implicit device-to-host transfer: {down}")
    # stages, each timed alone on the same inputs
    inp = hp._device_inputs(pos, cell, numbers, True)
    t_lists, _ = best(lambda: hp._device_inputs(pos, cell, numbers, True)["args"][7])
    t_blocks, (onb, offb, lists, _ok) = best(lambda: O._device(*inp["args"], inp["env"], inp["static"]))
    kd = jnp.asarray(kk, inp["dtype"])
    bloch = jax.jit(lambda o, f, l, k: (jax.vmap(lambda q: O._bloch_impl(o, f, l, q, 0, inp["n_orb"]))(k),
                                        jax.vmap(lambda q: O._bloch_impl(o, f, l, q, 1, inp["n_orb"]))(k)))
    t_bloch, (H, S) = best(lambda: bloch(onb, offb, lists, kd))
    herm = lambda M: 0.5 * (M + jnp.swapaxes(M.conj(), 1, 2))
    H, S = herm(H), herm(S)
    t_eig, _ = best(lambda: O._eigvals_k(H, S))
    t_geig, _ = best(lambda: jax.jit(jax.vmap(O._geig))(H, S))
    rest = t_full - t_lists - t_blocks - t_bloch - t_geig
    print(f"\n{N} atoms, {inp['n_orb']} orbitals, {len(kk)} k-points; E_band with forces and stress: {1e3 * t_full:.1f} ms ({1e3 * t_full / N:.2f} ms per atom); "
          f"E_band {float(out.energy):.4f} eV, max |F| {float(jnp.abs(out.gradient).max()):.3f} eV/A")
    for name, t in (("neighbour lists (three cutoffs, GPU)", t_lists), ("site features and all blocks", t_blocks), ("Bloch sums of H and S", t_bloch),
                    ("eigenvalues and eigenvectors (Cholesky + eigh)", t_geig), ("   eigenvalues only, for reference", t_eig),
                    ("the rest: density matrices and the reverse pass", rest)):
        print(f"   {name:52s} {1e3 * t:8.1f} ms  {100 * t / t_full:5.1f} %")
    calc = aj.ACECalculator(potential, ffi=True)
    b = a.copy()
    b.calc = calc

    def eff():
        b.positions = b.positions + 1e-9                      # invalidate ASE's cache
        return b.get_forces()
    t_eff, _ = best(eff)
    print(f"   E_eff, ACECalculator(ffi=True): kernel {calc.ffi}; {1e3 * t_eff:.2f} ms per energy and force call ({1e3 * t_eff / N:.3f} ms per atom)")


if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[3:])
    main(sys.argv[1], sys.argv[2], tuple(int(v) for v in kw.get("n", "1,2").split(",")), int(kw.get("nk", 2)))
