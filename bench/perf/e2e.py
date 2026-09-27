"""End-to-end calculator timing: ACECalculator (as benchmarked) vs the prototype
SkinDenseCalculator, optionally with a FastPACE model variant.

    PYTHONPATH=bench:src python bench/perf/e2e.py <yace> <system> <n> [--variant rec+pool+fm]

Each timed call displaces every atom by a fresh random vector of 1e-3 A from
the lattice positions (as consecutive MD steps would), so the skin calculator
takes its reuse path; the ASE calculator rebuilds its list every call either way.
Prints one JSON object.
"""
import argparse
import json
import statistics
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yace")
    ap.add_argument("system")
    ap.add_argument("n", type=int)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--skip-ase", action="store_true")
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    import jax.numpy as jnp
    import numpy as np
    from ase.calculators.calculator import all_changes

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from fast_calc import SkinDenseCalculator
    from ace_jax.calc.point import ACECalculator
    from ace_jax.eval.pace_model import load_yace
    from scaling.structures import supercell

    dt = getattr(jnp, args.dtype)
    at = supercell(args.system, args.n)
    pos0 = at.positions.copy()
    model, meta, _ = load_yace(args.yace, dtype=dt)
    if args.variant != "baseline":
        import variants
        model = variants.make(args.variant, model)
    rng = np.random.default_rng(1)
    out = {"yace": args.yace, "n": args.n, "dtype": args.dtype, "variant": args.variant,
           "device": str(jax.devices()[0].device_kind)}

    def run(calc, tag):
        def call():
            at.positions = pos0 + 1e-3 * rng.standard_normal(pos0.shape)
            calc.calculate(at, ["energy", "forces", "stress"], all_changes)
        t0 = time.perf_counter(); call(); first = time.perf_counter() - t0
        call()
        ts, splits = [], []
        for _ in range(args.reps):
            t0 = time.perf_counter(); call(); ts.append(time.perf_counter() - t0)
            splits.append(dict(calc.last_timing))
        med = statistics.median(ts)
        keys = [k for k in splits[0] if isinstance(splits[0][k], float)]
        out[tag] = {"call_s": med, "atoms_per_s": args.n / med, "first_s": first,
                    **{k: statistics.median(s[k] for s in splits) for k in keys}}
        return calc

    def fail(tag, ex):
        msg = repr(ex)
        out[tag] = {"status": "oom" if ("RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower())
                    else "error", "error": msg[:300]}

    # parity at one fixed geometry; the ASE calculator first, then freed
    geo = pos0 + 1e-3 * np.random.default_rng(7).standard_normal(pos0.shape)
    E_a = F_a = None
    if not args.skip_ase:
        try:
            ase_calc = ACECalculator(model, meta, dtype=dt, layout="dense")
            at.positions = geo
            ase_calc.calculate(at, ["energy", "forces", "stress"], all_changes)
            E_a, F_a = ase_calc.results["energy"], ase_calc.results["forces"].copy()
            run(ase_calc, "ase_calc")
        except Exception as ex:                                    # noqa: BLE001
            fail("ase_calc", ex)
        ase_calc = None
        import gc
        gc.collect()
    try:
        skin = SkinDenseCalculator(model, meta, dtype=dt)
        at.positions = geo
        skin.calculate(at, ["energy", "forces", "stress"], all_changes)
        if E_a is not None:
            out["parity_skin_vs_ase"] = {
                "dE_per_atom": abs(skin.results["energy"] - E_a) / args.n,
                "max_dF": float(np.max(np.abs(skin.results["forces"] - F_a)))}
        run(skin, "skin_calc")
        out["skin_calc"]["n_rebuilds"] = skin.n_rebuilds
        out["skin_calc"]["first_rebuild_s"] = skin.last_rebuild_s
        # steady-state rebuild (K_skin known, GPU neighbour matrix): what an MD
        # run pays each time an atom has moved skin/2
        ts = []
        for _ in range(3):
            skin._rebuild(at.positions, at.cell.array, at.pbc, at.numbers,
                          np.dtype(dt), K_cut=skin._nl["K"])
            ts.append(skin.last_rebuild_s)
        out["skin_calc"]["rebuild_s"] = statistics.median(ts)
        out["skin_calc"]["K"] = skin._nl["K"]
        out["skin_calc"]["K_skin"] = skin._nl["K_skin"]
    except Exception as ex:                                        # noqa: BLE001
        fail("skin_calc", ex)
    try:
        out["peak_bytes"] = jax.devices()[0].memory_stats().get("peak_bytes_in_use")
    except Exception:                                              # noqa: BLE001
        pass
    print(json.dumps(out))


if __name__ == "__main__":
    main()
