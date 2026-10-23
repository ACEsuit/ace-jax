"""Batched MD benchmark: µs per atom per step of JAX-MD's NVE under jax.vmap over B rattled Si cells
(`BatchCalculator.energy_fn`, `neighbour_lists`), the JAX model against the C++ kernel (ffi=True, CPU).
One JSON line per point.  The timed part is jitted chunks of --steps steps (lists fixed; skin large
enough that none is rebuilt in the run), median of --reps after a warm-up.

    python bench/batch/md.py --model <o4d12 model.npz> [--ffi] [--sizes 8 64 216] [--batch 1 16 64 256]"""
import argparse
import json
import platform
import statistics
import time

import jax

jax.config.update("jax_enable_x64", True)
from ase.build import bulk
from jax_md import simulate, space

from ace_jax import batch
from ace_jax.batch import BatchCalculator

CELLS = {2: lambda: bulk("Si", "diamond", 5.43), 8: lambda: bulk("Si", "diamond", 5.43, cubic=True),
         64: lambda: bulk("Si", "diamond", 5.43, cubic=True).repeat(2),
         216: lambda: bulk("Si", "diamond", 5.43, cubic=True).repeat(3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 64, 216])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64, 256])
    ap.add_argument("--ffi", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-atoms", type=int, default=2 ** 15)
    a = ap.parse_args()
    _, shift = space.free()
    for n in a.sizes:
        for B in a.batch:
            if B * n > a.max_atoms:
                continue
            ats = []
            for k in range(B):
                at = CELLS[n]()
                at.rattle(0.05, seed=k)
                ats.append(at)
            bc = BatchCalculator(a.model, ffi=a.ffi)
            nb = bc.neighbour_lists(ats, skin=1.0)
            nl = nb.allocate()
            init, apply = simulate.nve(bc.energy_fn(), shift, dt=1.0 / 10.18)
            p0 = batch.initial_momenta(jax.random.PRNGKey(0), nb.live, 0.01, 28.0855)
            state = jax.jit(jax.vmap(lambda R, p, nn, init=init: init(None, R, kT=0.0, mass=28.0855, momenta=p, neighbor=nn)))(
                nb.positions, p0, nl)
            step = jax.jit(jax.vmap(lambda s, nn, apply=apply: jax.lax.fori_loop(0, a.steps, lambda i, x: apply(x, neighbor=nn), s)))
            state = jax.block_until_ready(step(state, nl))
            ts = []
            for _ in range(a.reps):
                t0 = time.perf_counter()
                state = jax.block_until_ready(step(state, nl))
                ts.append(time.perf_counter() - t0)
            t = statistics.median(ts)
            print(json.dumps(dict(host=platform.node(), backend=jax.default_backend(), ffi=a.ffi, n=n, cells=B,
                                  steps=a.steps, s=t, us_per_atom_step=1e6 * t / (a.steps * n * B))), flush=True)


if __name__ == "__main__":
    main()
