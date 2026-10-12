"""Recursive (DAG) product basis vs the chain form: parity and CPU/GPU step time.

  python bench/aa_dag/run.py [--models ace/o4d16 pace/f800 ...] [--rep 3] [--atoms 1000]
    [--modes none prefix greedy] [--parity-only] [--out results.jsonl]

Models are the accuracy-cost Si fits (runs/accuracy_cost/{ace,pace}/<name>).  Each
configuration is a rattled Si diamond supercell; the step is ACECalculator's compiled
skin step (dense layout, lean, as deployed), timed over `--steps` calls with small
displacements that stay inside the skin.  Per mode: us/atom-step (median of --rep
blocks), compile s, and E/F/V errors against the chain form (mode none)."""
import argparse
import json
import os
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
from ase.build import bulk  # noqa: E402

from ace_jax.calc.point import ACECalculator  # noqa: E402
from ace_jax.eval import load  # noqa: E402
from ace_jax.eval.aa_dag import build_plan, plan_stats  # noqa: E402

ROOT = "/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/"


def path_of(name):
    return ROOT + name + ("/model.yace" if name.startswith("pace") else "/model.npz")


def structure(n_atoms, seed=0):
    rep = max(1, round((n_atoms / 8) ** (1 / 3)))
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((rep,) * 3)
    at.rattle(0.05, seed=seed)
    return at


def efv(calc, at):
    at = at.copy()
    at.calc = calc
    return at.get_potential_energy(), at.get_forces(), at.get_stress(voigt=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ace/o4d12", "ace/o4d16", "ace/o4d20", "pace/f800",
                                                    "pace/f3200"])
    ap.add_argument("--modes", nargs="+", default=["none", "prefix", "greedy"])
    ap.add_argument("--atoms", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--rep", type=int, default=3)
    ap.add_argument("--parity-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    dev = jax.devices()[0]
    for name in a.models:
        model, meta, *_ = load(path_of(name))
        ref = None
        for mode in a.modes:
            calc = ACECalculator(model, meta, aa_dag=None if mode == "none" else mode)
            m = calc.eval_model
            specs = getattr(m, "blk_aa_specs", None) or m.aa_specs
            st = (plan_stats(specs, build_plan(specs, 1 + max(int(np.asarray(s).max()) for s in specs), mode))
                  if mode != "none" else None)
            at = structure(216 if a.parity_only else a.atoms)
            t0 = time.perf_counter()
            E, F, V = efv(calc, at)
            t_first = time.perf_counter() - t0
            row = {"model": name, "mode": mode, "device": str(dev.platform), "host": os.uname().nodename,
                   "n_atoms": len(at), "first_call_s": t_first, "threads": os.environ.get("OMP_NUM_THREADS"),
                   "xla_flags": os.environ.get("XLA_FLAGS")}
            if st:
                row["plan"] = {k: v for k, v in st.items() if k != "per_level"}
            if ref is None:
                ref = (E, F, V)
            else:
                row["dE_rel"] = abs(E - ref[0]) / abs(ref[0])
                row["dF_rel_max"] = float(np.abs(F - ref[1]).max() / np.abs(ref[1]).max())
                row["dV_rel_max"] = float(np.abs(V - ref[2]).max() / np.abs(ref[2]).max())
            if not a.parity_only:
                rng = np.random.default_rng(1)
                pos0 = at.get_positions()
                ts = []
                for _ in range(a.rep):
                    t0 = time.perf_counter()
                    for _ in range(a.steps):
                        at.set_positions(pos0 + rng.normal(scale=0.01, size=pos0.shape))
                        at.calc = calc
                        at.get_forces()
                    ts.append((time.perf_counter() - t0) / a.steps)
                assert calc.last_timing["rebuilds"] <= 2, calc.last_timing
                row["step_s"] = float(np.median(ts))
                row["us_per_atom_step"] = 1e6 * row["step_s"] / len(at)
                row["model_s"] = calc.last_timing.get("model_s")
            print(json.dumps(row), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
            del calc


if __name__ == "__main__":
    main()
