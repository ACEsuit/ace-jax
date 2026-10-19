"""One JAX-MD case: an ace-jax model in a JAX-MD NVE run, the whole step on the device.

The LAMMPS protocol of run_lammps, so a row compares with its lammps rows: the
same rattled supercell, zero initial velocities, timestep 0.1 fs, skin 1.0 A
checked every step (JAX-MD `dr_threshold` = LAMMPS `neighbor 1.0 bin`,
`neigh_modify every 1 delay 0 check yes`: both rebuild once an atom has moved
skin / 2), `choose_steps` timed steps after a warm-up.  The timed segment is one
jitted `lax.fori_loop` over the steps -- neighbour-list update, model, NVE
update -- so step_s includes everything LAMMPS's Loop time does, and nothing
crosses to the host.

mode "jaxmd": JAX-MD's own cell list; "jaxmd-shift": ace_jax.jaxmd.neighbor_list, a
matscipy-neighbours list built on the device (its planned CUDA build: no host sync,
O(N K) memory, any cell), updated every step the same way.  The force is JAX-MD's own autodiff, `quantity.force` of `ace_jax.jaxmd`'s
energy (what `potential(...).force` is).  K, the model's slots per atom, is the LAMMPS bundles' exactly
(neighbour_capacity, margin 8): "skin" slots for rc + skin pairs, so no MD-time
overflow; the "-tight" modes ("cutoff" slots, as lammps-tight) for rc pairs, which
can overflow as atoms move -- energy and forces are then NaN, so a run that
overflowed ends "unstable", never truncated.  An overflow of the JAX-MD list is
retried once with a fresh list and recorded.

gate(row): JAX-MD E/F against ACECalculator on three rattled 216-atom cells --
not GATE_N 64: JAX-MD is minimum-image, so a box must exceed 2 (rcut + skin).
"""
import os
import time

import numpy as np

from scaling.structures import supercell

SKIN, DT_FS, MASS = 1.0, 0.1, {"Si": 28.0855, "Ge": 72.630}
TIME_UNIT_FS = 10.180505   # sqrt(amu A^2 / eV) in fs: JAX-MD in eV, A, amu
GATE_N, GATE_CELLS = 216, 3
CHUNK = 10                 # steps per compiled call (see run_case)
# parity.TOL["acejax"], or "acejax-splined" for a splined model: the same evaluation model, another list


def _chunk():
    """Rows per dense block: "auto" (jaxmd.block_rows, the lammps-jax bundles' rule) unless
    ACEJAX_AC_JAXMD_CHUNK fixes it."""
    c = os.environ.get("ACEJAX_AC_JAXMD_CHUNK", "auto")
    return c if c == "auto" else int(c)


def _setup(row, at, dtype, mode, K=None):
    import jax
    import jax.numpy as jnp
    from jax_md import partition, space
    from ace_jax import jaxmd
    dt = getattr(jnp, dtype)
    box = jnp.asarray(at.cell.array.T, dt)                 # JAX-MD: lattice vectors are columns
    disp, shift = space.periodic_general(box, fractional_coordinates=True)
    R = jnp.asarray(at.get_scaled_positions(), dt)
    kw = {"spline_tol": row["spline_tol"]} if "spline_tol" in row else {}
    rc = jaxmd.potential(row["path"], disp, at.numbers, dtype=dt, **kw).cutoff
    shift_list, tight = mode.startswith("jaxmd-shift"), mode.endswith("-tight")
    if shift_list:                                         # matscipy-neighbours on the device (ShiftList)
        nf = jaxmd.neighbor_list(box, rc, skin=SKIN, fractional_coordinates=True)
        nb, disp_k = nf.allocate(R), None
    else:                                                  # JAX-MD's own cell list
        nf = partition.neighbor_list(disp, box, rc, dr_threshold=SKIN, format=partition.Dense,
                                     fractional_coordinates=True)
        nb, disp_k = jaxmd.allocate(nf, R), disp
    if not K:   # the LAMMPS bundles' model slots exactly (run_lammps.capacity -> neighbour_capacity, margin 8):
        from ace_jax.export.lammps import neighbour_capacity   # "skin" (rc + skin pairs + 8: never overflows in MD)
        K = neighbour_capacity(at, rc, skin=SKIN, slots="cutoff" if tight else "skin", margin=8)["k_dense"]
    pot = jaxmd.potential(row["path"], disp_k, at.numbers, K=K, dtype=dt,
                          chunk=_chunk(), **kw)
    force = pot.force                                     # -grad(energy): JAX-MD's quantity.force
    return jax, jnp, pot, force, nf, nb, R, shift, K


def run_case(row, n_atoms, dtype, device, prev=None, mode="jaxmd"):
    import jax
    from jax_md import simulate
    from scaling.run_lammps import choose_steps
    out = {"code": row["code"], "mode": mode, "model": row["name"], "size": row["size"],
           "system": row["system"], "n_atoms": n_atoms, "device": device, "dtype": dtype,
           "status": "ok", "skin": SKIN, "dt_fs": DT_FS, "chunk": _chunk()}
    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    at = supercell(row["system"], n_atoms, rattle=row.get("rattle", 0.0))
    from ace_jax import jaxmd
    rc = float(jaxmd.eval_model(row["path"])[2])
    if not mode.startswith("jaxmd-shift") and not jaxmd.min_image_ok(at.cell.array, rc + SKIN):   # JAX-MD misses images
        return {**out, "status": "unsupported", "error": f"cell narrower than 2 (rcut + skin) = {2 * (rc + SKIN):.2f} A"}
    t0 = time.perf_counter()
    jax_, jnp, pot, force, nf, nb, R, shift, K = _setup(row, at, dtype, mode)
    masses = jnp.asarray([MASS[s] for s in at.get_chemical_symbols()], R.dtype)[:, None]
    init, apply = simulate.nve(force, shift, DT_FS / TIME_UNIT_FS)
    st = init(jax.random.PRNGKey(0), R, mass=masses, kT=0.0, neighbor=nb)   # kT 0: zero velocities, as LAMMPS
    st = st.set(momentum=jnp.zeros_like(st.momentum))

    def run(st, nb, n):
        def body(i, c):                                    # k: steps whose update rebuilt the list
            s, b, k = c
            b2 = b.update(s.position)
            ref = (lambda x: x.ref_frac) if mode.startswith("jaxmd-shift") else (lambda x: x.reference_position)
            k = k + jnp.any(ref(b2) != ref(b)).astype(jnp.int32)
            return apply(s, neighbor=b2), b2, k
        return jax.lax.fori_loop(0, n, body, (st, nb, jnp.zeros((), jnp.int32)))
    # One static chunk of CHUNK steps, compiled by the warm-up and called repeatedly: a static trip count
    # lets XLA keep the loop body a CUDA graph (a traced count measured 2.2x slower per step at 512 atoms),
    # and one executable means no compilation inside the timing (a timed n differing from the warm-up's
    # compiled again: ~6 ms/step at 200 steps on an A100).  Step counts are rounded to whole chunks.
    jrun = jax.jit(run, static_argnums=2)

    def run(st, nb, n):
        k = jnp.zeros((), jnp.int32)
        for _ in range(n // CHUNK):
            st, nb, b = jrun(st, nb, CHUNK)
            k = k + b
        return st, nb, k
    steps, warmup = choose_steps((prev or {}).get("step_s"), (prev or {}).get("n_atoms"), n_atoms)
    steps, warmup = CHUNK * max(1, round(steps / CHUNK)), CHUNK * max(1, round(warmup / CHUNK))
    retries = 0
    for n, timed in ((warmup, False), (steps, True)):
        while True:
            jax.block_until_ready(st.position)
            t1 = time.perf_counter()
            s2, nb2, builds = run(st, nb, n)
            jax.block_until_ready(s2.position)
            el = time.perf_counter() - t1
            if not bool(nb2.did_buffer_overflow):
                break
            if retries:                                    # once: a fresh list sized at this structure
                out.update(status="overflow", error="JAX-MD neighbour list overflowed twice")
                return out
            retries += 1
            from ace_jax import jaxmd
            nb = nf.allocate(st.position) if mode.startswith("jaxmd-shift") else jaxmd.allocate(nf, st.position)
            jax.block_until_ready(run(st, nb, CHUNK)[0].position)   # compile outside the timing
        st, nb = s2, nb2
        if not timed:
            out["setup_s"] = time.perf_counter() - t0      # list, model, compile and warm-up
    F = st.force if hasattr(st, "force") else None
    nan = not bool(jnp.all(jnp.isfinite(st.position))) or (F is not None and not bool(jnp.all(jnp.isfinite(F))))
    out.update(steps=steps, warmup=warmup, loop_s=el, step_s=el / steps, K=K, K_list=int(nb.idx.shape[1]),
               neighbor_builds=int(builds),                # timed segment, as LAMMPS's "Neighbor list builds"
               list_retries=retries, nan=nan, atom_steps_per_s=n_atoms * steps / el)
    if nan:
        out["status"] = "unstable"
    try:
        out["gpu_peak_jax_bytes"] = jax.devices()[0].memory_stats().get("peak_bytes_in_use")
    except Exception:                                      # CPU backends report no stats
        pass
    return out


def gate(row, mode="jaxmd"):
    """{status, checks, max_dE_per_atom, max_dF}: JAX-MD vs ACECalculator, float64."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from scaling import parity
    from ace_jax.calc.point import ACECalculator
    checks = []
    for k in range(GATE_CELLS):
        at = supercell(row["system"], GATE_N, seed=k, rattle=row.get("rattle", 0.0))
        chk = {"cell": k}
        try:
            calc = ACECalculator(row["path"], **parity._tol_kw(row))
            at.calc = calc
            E0, F0 = at.get_potential_energy(), at.get_forces()
            _, jnp, pot, force, nf, nb, R, _, K = _setup(row, at, "float64", mode)
            E1, F1 = float(jax.jit(pot.energy)(R, nb)), np.asarray(jax.jit(force)(R, nb))
            tol = parity.TOL["acejax-splined" if "spline_tol" in row else "acejax"]
            chk.update(K=K, tol=list(tol), **parity.compare(E0, F0, E1, F1, len(at), tol))
        except Exception as ex:
            chk.update(status="error", error=parity.error_summary(ex))
        checks.append(chk)
    st = {c["status"] for c in checks}
    status = "parity_ok" if st == {"parity_ok"} else "parity_fail" if "parity_fail" in st else "error"
    dE = [c["dE_per_atom"] for c in checks if "dE_per_atom" in c]
    dF = [c["max_dF"] for c in checks if "max_dF" in c]
    return {"gate": mode, "n_atoms": GATE_N, "cells": GATE_CELLS, "status": status, "checks": checks,
            "max_dE_per_atom": max(dE, default=None), "max_dF": max(dF, default=None)}
