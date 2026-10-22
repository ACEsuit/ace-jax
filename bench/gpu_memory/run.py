"""GPU memory and time of the ace-jax force call, per variant (docs/dev/gpu-memory-results.md).

    cd bench && python -m gpu_memory.run sweep  --model M.npz [--variant NAME --calc-kw JSON] [--out R.jsonl]
    cd bench && python -m gpu_memory.run bisect --model M.npz [...]       # largest bulk cell that runs
    cd bench && python -m gpu_memory.run check  --model M.npz --calc-kw JSON   # parity, FD, NVE vs the default

`sweep`: every (cell, size, mode) in its own process (`peak_bytes_in_use` is
per process, and an OOM must not end the sweep), as deployed: `ACECalculator`
(skin list, dense layout, lean, aa_dag auto) with the variant's keyword
arguments.  Cells: diamond Si `bulk`; a `slab` 8 cubic cells (43 A) thick with
20 A of vacuum; a `crack`, bulk with a 4 A slit through half the cell.  All
rattled by 0.02 A.  Modes: `ef` the calculator call (E, F, virial), `e` an
energy-only step over the same skin list.  Per case: `peak_bytes_in_use` after
the neighbour-list build and after the timed calls (XLA_PYTHON_CLIENT_PREALLOCATE
false); the compiled step's `memory_analysis()` (arguments, temp, outputs);
with --dump, the temp attributed to stages (`gpu_memory.attribute`) and a
device memory profile; and the median of --reps MD-like calls (each displaces
every atom by N(0, 1e-3 A) first, so the skin list is reused), as us/atom-step.

`bisect`: the largest bulk cell whose E+F call runs, by bisection on the
supercell repeat (doubling up from the largest swept size first).

`check`: the variant against the default calculator on rattled bulk cells and
the crack and slab (E, F, stress relative error), central finite differences of its
forces on a 64-atom cell, and NVE energy drift (velocity Verlet, 1 fs, 300 K)
for both.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

SIZES = (8000, 32768, 64000, 97336)
CELLS = ("bulk", "slab", "crack")
A_SI = 5.43
RESULT = "RESULT "


def repeats(kind, n):
    """Supercell repeats of the 8-atom cubic cell giving about n atoms."""
    if kind == "slab":
        r = max(1, round((n / 64) ** 0.5))
        return (r, r, 8)
    r = max(1, round((n / 8) ** (1 / 3)))
    return (r, r, r)


def cell(kind, n=None, reps=None, seed=0, rattle=0.02):
    import numpy as np
    from ase.build import bulk
    at = bulk("Si", "diamond", A_SI, cubic=True) * (reps or repeats(kind, n))
    if kind == "slab":
        at.cell[2, 2] += 20.0
    elif kind == "crack":           # a 4 A slit, normal to y, through half of x
        p, L = at.positions, at.cell.lengths()
        del at[(np.abs(p[:, 1] - L[1] / 2) < 2.0) & (p[:, 0] < L[0] / 2)]
    elif kind != "bulk":
        raise ValueError(f"cell must be one of {CELLS}, got {kind!r}")
    at.rattle(rattle, seed=seed)
    return at


# ------------------------------------------------------------------ child: one case
def _energy_step(static, rc, chunk):
    """Energy only, over the same skin list and compaction as `calc.skin.step`."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from ace_jax.calc import skin as skin_list
    from ace_jax.eval.edge_model import total_energy

    def energy(params, u, arrays, K):
        with jax.default_matmul_precision("highest"):
            model = eqx.combine(params, static)
            idx_s, rij0_s, live_s, _rev, node_z, _h = arrays
            rij_s = rij0_s + (u[idx_s] - u[:, None, :])
            valid_s = live_s & (jnp.sum(rij_s * rij_s, axis=-1) < rc * rc)
            order, mask, _ = skin_list.compact(valid_s, K)
            idx = jnp.where(mask, jnp.take_along_axis(idx_s, order, axis=1), 0)
            pad = jnp.asarray([1.0, 0.0, 0.0], u.dtype) * model.pad_cutoff()
            rij = jnp.where(mask[..., None], jnp.take_along_axis(rij_s, order[..., None], axis=1), pad)
            zi = jnp.broadcast_to(node_z[:, None], idx.shape)
            return total_energy(model.site_energies_dense_blocked(rij, zi, node_z[idx], mask, node_z,
                                                                  chunk))
    return jax.jit(energy, static_argnames=("K",))


def _patch(chunk=None, aa_chunk=None):
    """Harness-only experiments, for this process: `chunk` the dense row-block size
    (`energy_forces_virial_dense`'s default, `default_chunk()`); `aa_chunk` the DAG product
    basis in checkpointed column chunks on every backend (`dag_contract`'s chunk=int;
    "auto" chunks on the CPU only).  A calculator compiled before the patch keeps the
    old code (its step is traced on first use)."""
    import functools

    from ace_jax.eval import aa_dag, edge_model, model
    if chunk:
        f = edge_model.EdgeSiteModel.energy_forces_virial_dense
        edge_model.EdgeSiteModel.energy_forces_virial_dense = functools.partialmethod(f, chunk=chunk)
    if aa_chunk:
        model.dag_contract = functools.partial(aa_dag.dag_contract, chunk=aa_chunk)


def measure(case):
    import jax
    jax.config.update("jax_enable_x64", True)
    import equinox as eqx
    import numpy as np

    from ace_jax.calc import ACECalculator
    from ace_jax.eval.edge_model import default_chunk

    _patch(case.get("chunk"), case.get("aa_chunk"))
    dev = jax.devices()[0]
    stats = lambda: dev.memory_stats() or {}
    at = cell(case["cell"], case.get("n"), case.get("rep"))
    n = len(at)
    calc = ACECalculator(case["model"], **case.get("calc_kw", {}))
    at.calc = calc
    row = {k: case.get(k) for k in ("variant", "cell", "mode", "calc_kw", "chunk", "aa_chunk")}
    row.update(n=n, device=dev.device_kind, jax=jax.__version__)
    t0 = time.perf_counter()
    st, _ = calc._skin_build(at.positions, at.cell.array, at.pbc, at.numbers, np.dtype("float64"), None)
    row.update(nlist_s=time.perf_counter() - t0, K=st.K, K_skin=st.K_skin, n_edges=st.n_edges,
               fill=st.n_edges / (n * st.K), bytes_limit=stats().get("bytes_limit"),
               peak_after_nlist=stats().get("peak_bytes_in_use"))
    u0 = jax.numpy.zeros((n, 3))
    if case["mode"] == "ef":
        t0 = time.perf_counter()
        at.get_forces()
        row["first_call_s"] = time.perf_counter() - t0
        row["layout"] = calc.last_layout
        if calc.last_layout != "dense":
            raise RuntimeError(f"layout {calc.last_layout}: the harness measures the dense skin step")
        params, static = eqx.partition(calc.eval_model, eqx.is_array)
        lowered = calc._skin_jit[2].lower(params, u0, st.arrays, K=st.K)

        def call():
            at.positions = at.positions + rng.normal(0, 1e-3, (n, 3))
            at.get_forces()
    else:
        params, static = eqx.partition(calc.eval_model, eqx.is_array)
        f = _energy_step(static, calc.cutoff, case.get("chunk") or default_chunk())
        t0 = time.perf_counter()
        jax.block_until_ready(f(params, u0, st.arrays, K=st.K))
        row["first_call_s"] = time.perf_counter() - t0
        lowered = f.lower(params, u0, st.arrays, K=st.K)

        def call():
            u = jax.device_put(st.displacements(at.positions + rng.normal(0, 1e-3, (n, 3)), np.float64))
            jax.block_until_ready(f(params, u, st.arrays, K=st.K))
    row["peak_after_warmup"] = stats().get("peak_bytes_in_use")
    row["in_use_after_warmup"] = stats().get("bytes_in_use")
    m = lowered.compile().memory_analysis()
    row.update(arg_bytes=m.argument_size_in_bytes, temp_bytes=m.temp_size_in_bytes,
               out_bytes=m.output_size_in_bytes)
    # the program's own peak: what is resident when it starts, plus its temp and outputs (the
    # process peak also holds compile-time autotuning scratch)
    row["program_peak"] = row["in_use_after_warmup"] + m.temp_size_in_bytes + m.output_size_in_bytes
    rng = np.random.default_rng(0)
    times = []
    for _ in range(case.get("calls", 20)):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    row.update(call_s=statistics.median(times), us_per_atom_step=1e6 * statistics.median(times) / n,
               peak_bytes=stats().get("peak_bytes_in_use"))
    row["peak_bytes_per_atom"] = row["peak_bytes"] / n if row["peak_bytes"] else None
    if case["mode"] == "ef":
        row["model_s"] = calc.last_timing["model_s"]
        row["rebuilds"] = calc.last_timing["rebuilds"]
    if case.get("dump"):
        from gpu_memory.attribute import attribute
        att = attribute(case["dump"])
        row["attribution"] = {k: att[k] for k in ("stages", "arguments", "slack", "peak_live", "peak_time")}
        row["top"] = att["top"][:12]
        jax.profiler.save_device_memory_profile(os.path.join(case["dump"], "device_memory.prof"))
    return row


def child_main():
    case = json.loads(sys.argv[2])
    print(RESULT + json.dumps(measure(case)), flush=True)


# ------------------------------------------------------------------ parent: one subprocess per case
def run_case(case, timeout=3000):
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    if case.get("dump"):
        os.makedirs(case["dump"], exist_ok=True)
        env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + f" --xla_dump_to={case['dump']} --xla_dump_hlo_as_text"
                            " --xla_dump_hlo_module_re=jit_(packed|energy)").strip()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(p for p in (here, env.get("PYTHONPATH")) if p)
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, "-m", "gpu_memory.run", "_child", json.dumps(case)],
                           env=env, capture_output=True, text=True, timeout=timeout)
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out, err, rc = e.stdout or "", (e.stderr or "") + "\nTIMEOUT", -1
        out, err = (out.decode() if isinstance(out, bytes) else out), (err.decode() if isinstance(err, bytes) else err)
    for line in out.splitlines():
        if line.startswith(RESULT):
            return json.loads(line[len(RESULT):])
    oom = any(s in err for s in ("RESOURCE_EXHAUSTED", "out of memory", "Out of memory", "OOM"))
    tail = "\n".join(err.strip().splitlines()[-6:])
    print(f"  FAILED rc={rc} oom={oom}: {tail}", file=sys.stderr, flush=True)
    return {**{k: case.get(k) for k in ("variant", "cell", "mode", "calc_kw", "chunk", "aa_chunk", "n", "rep")},
            "error": tail[-2000:], "oom": oom, "rc": rc, "wall_s": time.perf_counter() - t0}


def _append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _show(r):
    if "error" in r:
        return f"{r.get('cell')} n~{r.get('n')} {r.get('mode')}: {'OOM' if r['oom'] else 'error'}"
    return (f"{r['cell']} n={r['n']} {r['mode']}: peak {r['peak_bytes'] / 1e9:.2f} GB "
            f"({r['peak_bytes_per_atom'] / 1e3:.1f} KB/atom), temp {r['temp_bytes'] / 1e9:.2f} GB, "
            f"{r['us_per_atom_step']:.3f} us/atom-step")


def sweep(a):
    for c in a.cells:
        for n in a.sizes:
            for mode in a.modes:
                dump = (os.path.join(a.dump_dir, f"{a.variant}-{c}-{n}-{mode}")
                        if a.dump_dir and n in a.dump_sizes else None)
                case = dict(model=a.model, variant=a.variant, cell=c, n=n, mode=mode, calc_kw=a.calc_kw,
                            calls=a.reps, dump=dump, chunk=a.chunk, aa_chunk=a.aa_chunk)
                r = run_case(case)
                _append(a.out, r)
                print(_show(r), flush=True)


def bisect(a):
    """Largest r (bulk r x r x r cubic cells, 8 r^3 atoms) whose E+F call runs."""
    def ok(r):
        row = run_case(dict(model=a.model, variant=a.variant, cell="bulk", rep=(r, r, r), n=8 * r ** 3,
                            mode="ef", calc_kw=a.calc_kw, calls=2, chunk=a.chunk, aa_chunk=a.aa_chunk), timeout=a.timeout)
        row["bisect"] = True
        _append(a.out, row)
        print(f"  r={r} ({8 * r ** 3} atoms): " + _show(row), flush=True)
        return "error" not in row
    lo, hi = a.r_lo, None
    r = a.r_start or lo
    while hi is None:                                  # grow by ~2x atoms until it fails
        if ok(r):
            lo, r = r, max(r + 1, round(r * 2 ** (1 / 3)))
        else:
            hi = r
        if r > a.r_max:
            break
    while hi is not None and hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    print(json.dumps({"variant": a.variant, "max_r": lo, "max_atoms": 8 * lo ** 3, "first_fail_r": hi}))
    _append(a.out, {"variant": a.variant, "bisect_result": True, "max_r": lo, "max_atoms": 8 * lo ** 3,
                    "first_fail_r": hi, "first_fail_atoms": 8 * hi ** 3 if hi else None})


# ------------------------------------------------------------------ check: parity, FD, NVE
def _efs(calc, at):
    at = at.copy()
    at.calc = calc
    return at.get_potential_energy(), at.get_forces(), at.get_stress(voigt=False)


def _nve(calc, at, steps, every=10):
    import numpy as np
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    at = at.copy()
    at.calc = calc
    MaxwellBoltzmannDistribution(at, temperature_K=300, rng=np.random.default_rng(1))
    md = VelocityVerlet(at, timestep=1.0 * units.fs)
    etot, t0 = [at.get_total_energy()], time.perf_counter()
    for _ in range(steps // every):
        md.run(every)
        etot.append(at.get_total_energy())
    e = (np.array(etot) - etot[0]) / len(at)
    slope = np.polyfit(np.arange(len(e)) * every, e, 1)[0]
    return {"steps": steps, "n": len(at), "drift_eV_atom": float(e[-1]), "max_dev_eV_atom": float(np.abs(e).max()),
            "slope_eV_atom_per_ps": float(slope * 1000), "wall_s": time.perf_counter() - t0}


def check(a):
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np

    from ace_jax.calc import ACECalculator
    out = {"variant": a.variant, "calc_kw": a.calc_kw, "chunk": a.chunk, "aa_chunk": a.aa_chunk, "check": True,
           "parity": []}
    cases = [("bulk", (6, 6, 6), s, 0.05) for s in range(3)] + [("crack", (8, 8, 8), 0, 0.05),
                                                                 ("slab", (6, 6, 8), 0, 0.05),
                                                                 ("bulk", (10, 10, 10), 4, 0.05)]  # > 1 row block at chunk <= 4096
    cells = [cell(kind, reps=reps, seed=seed, rattle=rat) for kind, reps, seed, rat in cases]
    base = ACECalculator(a.model)                    # the default, evaluated before any patch
    ref = [_efs(base, at) for at in cells]
    md_cell = cell("bulk", reps=(8, 8, 8), seed=0, rattle=0.0)
    if a.nve_steps and a.nve_baseline:
        out["nve_baseline"] = _nve(base, md_cell, a.nve_steps)
    _patch(a.chunk, a.aa_chunk)
    var = ACECalculator(a.model, **a.calc_kw)
    for (kind, _, seed, _), at, (E0, F0, S0) in zip(cases, cells, ref):
        E1, F1, S1 = _efs(var, at)
        out["parity"].append({"cell": kind, "n": len(at), "seed": seed,
                              "E_rel": abs(E1 - E0) / abs(E0),
                              "F_rel": float(np.abs(F1 - F0).max() / np.abs(F0).max()),
                              "S_rel": float(np.abs(S1 - S0).max() / np.abs(S0).max())})
    out["parity_max_rel"] = max(max(p["E_rel"], p["F_rel"], p["S_rel"]) for p in out["parity"])
    at = cell("bulk", reps=(2, 2, 2), seed=3, rattle=0.05)
    at.calc = var
    F = at.get_forces()
    rng, h, err = np.random.default_rng(0), 1e-4, 0.0
    for i, k in zip(rng.choice(len(at), 6, replace=False), rng.integers(0, 3, 6)):
        e = []
        for s in (h, -h):
            b = at.copy()
            b.calc = var
            b.positions[i, k] += s
            e.append(b.get_potential_energy())
        err = max(err, abs(-(e[0] - e[1]) / (2 * h) - F[i, k]))
    out["fd_max_abs_err"], out["fd_rel_err"] = err, err / np.abs(F).max()
    if a.nve_steps:
        out["nve"] = _nve(var, md_cell, a.nve_steps)
    _append(a.out, out)
    print(json.dumps(out, indent=1))


def main(argv=None):
    if len(sys.argv) > 1 and sys.argv[1] == "_child":
        return child_main()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("sweep", "bisect", "check"))
    p.add_argument("--model", required=True)
    p.add_argument("--variant", default="baseline")
    p.add_argument("--calc-kw", type=json.loads, default={}, help="ACECalculator keyword arguments (JSON)")
    p.add_argument("--out", default="gpu_memory/results/results.jsonl")
    p.add_argument("--cells", type=lambda s: s.split(","), default=list(CELLS))
    p.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=list(SIZES))
    p.add_argument("--modes", type=lambda s: s.split(","), default=["ef", "e"])
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--chunk", type=int, help="experiment: dense row-block size (default: default_chunk())")
    p.add_argument("--aa-chunk", type=int, help="experiment: DAG product-basis column chunk on the GPU too")
    p.add_argument("--dump-dir", help="XLA dumps + attribution for --dump-sizes")
    p.add_argument("--dump-sizes", type=lambda s: [int(x) for x in s.split(",")], default=[32768])
    p.add_argument("--r-lo", type=int, default=23, help="bisect: a repeat known to fit")
    p.add_argument("--r-start", type=int)
    p.add_argument("--r-max", type=int, default=200)
    p.add_argument("--timeout", type=int, default=1500)
    p.add_argument("--nve-steps", type=int, default=1000)
    p.add_argument("--nve-baseline", action="store_true")
    a = p.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    {"sweep": sweep, "bisect": bisect, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
