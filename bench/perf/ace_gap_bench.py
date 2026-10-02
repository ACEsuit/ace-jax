"""ACE vs PACE on one GPU: the dense E/F/V model call (and optionally the whole
ACECalculator call) of size-matched ACE (.npz) and PACE (.yace) models, plus the
`ace_fast` prototype variants of the ACE model, interleaved round by round so
container-to-container GPU variation cancels (docs/dev/ace-vs-pace-gap.md).

    PYTHONPATH=bench:src python bench/perf/ace_gap_bench.py <system>_<size> <n> \
        [--variants prune,prune+lblock,...] [--pace-variants sbessel_mm] \
        [--rounds 5] [--reps 10] [--calc]

Prints one JSON object: per case the median over rounds of the per-round median
efv time, atom-steps/s, parity of each variant against the stock ACE model, and
(with --calc) the ACECalculator end-to-end call time (MD-like, skin list).
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODELS = os.path.join(ROOT, "bench", "scaling", "models")


def dense_args(meta, system, n, dt):
    import jax.numpy as jnp
    import numpy as np
    from ase.neighborlist import neighbor_list

    from ace_jax.eval.nlist import dense_graph
    from scaling.structures import supercell
    at = supercell(system, n)
    rc = float(meta["rcut"])
    i = neighbor_list("i", at, rc)
    K = int(np.bincount(i, minlength=len(at)).max())
    g = dense_graph(at.get_positions(), at.cell.array, at.pbc, rc, K)
    z2i = {int(z): k for k, z in enumerate(meta["elements"])}
    node_z = jnp.asarray(np.array([z2i[int(z)] for z in at.get_atomic_numbers()], np.int32))
    idx = jnp.asarray(g.idx, jnp.int32)
    mask = jnp.arange(idx.shape[1])[None, :] < jnp.asarray(g.count, jnp.int32)[:, None]
    return at, (jnp.asarray(g.rij, dt), jnp.broadcast_to(node_z[:, None], idx.shape),
                node_z[idx], idx, mask, node_z)


def timed(fn, reps):
    import jax
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def calc_time(model, meta, at, dt, reps):
    """ACECalculator call, MD-like (as bench/perf/microbench.py): median s."""
    import numpy as np
    from ase.calculators.calculator import all_changes

    from ace_jax.calc.point import ACECalculator
    calc = ACECalculator(model, meta, dtype=dt)
    at = at.copy()
    call = lambda: calc.calculate(at, ["energy", "forces", "stress"], all_changes)
    call()
    rng = np.random.default_rng(0)
    ts = []
    for i in range(reps + 1):
        at.positions += rng.normal(0, 1e-3, at.positions.shape)
        t0 = time.perf_counter(); call(); d = time.perf_counter() - t0
        if i:
            ts.append(d)
    return statistics.median(ts), calc.last_layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="e.g. Cantor_medium")
    ap.add_argument("n", type=int)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--variants", default="prune,prune+lblock,compact,compact+pairfold")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-pace", action="store_true")
    ap.add_argument("--pace-variants", default="", help="pace_fast variants, e.g. sbessel_mm")
    ap.add_argument("--calc", action="store_true", help="also time the ACECalculator call")
    a = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", a.dtype == "float64")
    import jax.numpy as jnp
    sys.path.insert(0, HERE)
    import ace_fast
    import pace_fast

    from ace_jax.eval.io import load
    from ace_jax.eval.model import highest_precision
    dt = getattr(jnp, a.dtype)
    system = a.model.split("_")[0]
    ace, meta_a, _ = load(os.path.join(MODELS, f"ace_{a.model}.npz"), dtype=dt)
    at, args = dense_args(meta_a, system, a.n, dt)
    cases = {"ace": (ace, args)}
    out = {"model": a.model, "n": a.n, "dtype": a.dtype, "K": int(args[4].shape[1]),
           "device": str(jax.devices()[0].device_kind), "jax": jax.__version__,
           "parity": {}, "cases": {}}
    for v in [v for v in a.variants.split(",") if v]:
        cases[f"ace:{v}"] = (ace_fast.make(v, ace), args)
    if not a.no_pace:
        pace, meta_p, _ = load(os.path.join(MODELS, f"pace_{a.model}.yace"), dtype=dt)
        _, pargs = dense_args(meta_p, system, a.n, dt)
        cases["pace"] = (pace, pargs)
        for v in [v for v in a.pace_variants.split(",") if v]:
            cases[f"pace:{v}"] = (pace_fast.make(v, pace), pargs)
    efv = jax.jit(lambda mm, *x: mm.energy_forces_virial_dense(*x))
    with highest_precision():
        for k, (m, x) in cases.items():
            if k.startswith("ace:"):
                out["parity"][k] = ace_fast.check(ace, m, x)
            if k.startswith("pace:"):
                out["parity"][k] = ace_fast.check(cases["pace"][0], m, x)
            t0 = time.perf_counter()
            jax.block_until_ready(efv(m, *x))
            out["cases"][k] = {"compile_s": time.perf_counter() - t0, "rounds": []}
        for rnd in range(a.rounds):
            order = list(cases) if rnd % 2 == 0 else list(cases)[::-1]
            for k in order:
                m, x = cases[k]
                out["cases"][k]["rounds"].append(timed(lambda: efv(m, *x), a.reps))  # noqa: B023
        for c in out["cases"].values():
            c["efv_s"] = statistics.median(c["rounds"])
            c["atom_steps_per_s"] = a.n / c["efv_s"]
            c["vs_ace"] = out["cases"]["ace"]["efv_s"] / c["efv_s"] if "ace" in out["cases"] else None
        if a.calc:
            for k, (m, _) in cases.items():
                meta = meta_a if k.startswith("ace") else meta_p
                s, lay = calc_time(m, meta, at, dt, a.reps)
                out["cases"][k].update(calc_s=s, calc_atom_steps_per_s=a.n / s, layout=lay)
    try:
        out["peak_bytes"] = jax.devices()[0].memory_stats().get("peak_bytes_in_use")
    except Exception:                                              # noqa: BLE001
        pass
    print(json.dumps(out))


if __name__ == "__main__":
    main()
