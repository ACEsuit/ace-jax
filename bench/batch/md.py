"""Batched MD benchmark: µs per atom per step of JAX-MD's NVE under jax.vmap over B rattled Si cells
(`BatchNeighbours.energy_fn`: compacted K slots, batch-sized row blocks), the JAX model against the C++ kernel (ffi=True, CPU).
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
    ap.add_argument("--update", action="store_true",
                    help="check the skin (and rebuild) every step inside the loop (BatchNeighbours.update_fn)")
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
            init, apply = simulate.nve(nb.energy_fn(nl), shift, dt=1.0 / 10.18)   # compacted K, batch-sized blocks
            p0 = batch.initial_momenta(jax.random.PRNGKey(0), nb.live, 0.01, 28.0855,
                                    dtype=nb.positions.dtype)
            state = jax.jit(jax.vmap(lambda R, p, nn, init=init: init(None, R, kT=0.0, mass=28.0855, momenta=p, neighbor=nn)))(
                nb.positions, p0, nl)
            if a.update:
                update = nb.update_fn()

                def body(i, c, apply=apply, update=update):
                    st, nn, k = c
                    nn, rebuilt, _ = update(st.position, nn)
                    return jax.vmap(lambda x, m: apply(x, neighbor=m))(st, nn), nn, k + rebuilt
                step = jax.jit(lambda st, nn: jax.lax.fori_loop(0, a.steps, body, (st, nn, 0)))
            else:
                step = jax.jit(jax.vmap(lambda s, nn, apply=apply: jax.lax.fori_loop(
                    0, a.steps, lambda i, x: apply(x, neighbor=nn), s)))
            run = ((lambda st, nn, step=step: step(st, nn)[:2]) if a.update
                   else (lambda st, nn, step=step: (step(st, nn), nn)))
            state, nl = jax.block_until_ready(run(state, nl))
            ts = []
            for _ in range(a.reps):
                t0 = time.perf_counter()
                state, nl = jax.block_until_ready(run(state, nl))
                ts.append(time.perf_counter() - t0)
            t = statistics.median(ts)
            print(json.dumps(dict(host=platform.node(), backend=jax.default_backend(), ffi=a.ffi, update=a.update, n=n, cells=B,
                                  steps=a.steps, s=t, us_per_atom_step=1e6 * t / (a.steps * n * B))), flush=True)


if __name__ == "__main__":
    main()
