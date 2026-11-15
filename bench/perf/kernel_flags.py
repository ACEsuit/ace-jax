"""The C++ kernel's time by compile flags (docs/dev/kernel-wheels.md): what a prebuilt wheel gives up against
-march=native.  One child process per (flags, model, threads); the rounds interleave the flag sets.  Pin it
on a many-core node (taskset -c 0-15): unpinned 16-thread runs on a 128-core node differed by up to 1.5x.
usage: kernel_flags.py REPO OUT.jsonl [rounds]   (env KB_FLAGS: ';'-separated flag sets, KB_THREADS: e.g. 1,16)"""
import json, os, subprocess, sys, pathlib

CHILD = r'''
import json, os, sys, time, numpy as np, jax
jax.config.update("jax_enable_x64", True)
from ase.build import bulk
from ace_jax import ACECalculator
from ace_jax.eval import ffi
path, threads, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
at = bulk("Si", "diamond", a=5.43, cubic=True) * (n, n, n)
rng = np.random.default_rng(0); at.rattle(0.05, seed=0); p0 = at.positions.copy()
t0 = time.perf_counter(); lib = ffi.library(); tb = time.perf_counter() - t0
calc = ACECalculator(path, ffi=True, ffi_threads=threads, skin=1.0); at.calc = calc
ts = []; f = None
for k in range(14):
    at.positions = p0 + rng.normal(scale=0.01, size=p0.shape)      # inside the skin: no list rebuild
    f = at.get_forces(); e = at.get_potential_energy()
    ts.append(calc.last_timing["model_s"])
ts = np.array(ts[4:])
print("KB " + json.dumps(dict(model_ms=float(np.median(ts)) * 1e3, min_ms=float(ts.min()) * 1e3, natoms=len(at),
      build_s=tb, ffi=calc.ffi, e=float(e), fmax=float(np.abs(f).max()), lib_bytes=os.path.getsize(lib))))
'''

repo, out = pathlib.Path(sys.argv[1]).resolve(), sys.argv[2]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
flagsets = os.environ.get("KB_FLAGS", "-O3 -march=native").split(";")
threads = [int(t) for t in os.environ.get("KB_THREADS", "1").split(",")]
models = [("si_fitted", 8), ("si_m710", 6), ("si_l2849", 5)]       # o3d10 (4096 atoms), o4d14 (1728), o4d18 (1000)
py = sys.executable
for r in range(rounds):
    for name, n in models:
        for th in threads:
            for fl in flagsets:
                env = dict(os.environ, ACEJAX_FFI_FLAGS=fl, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
                           ACEJAX_FFI_CACHE=str(pathlib.Path(out).resolve().parent / "ffi-cache"),
                           PYTHONPATH=str(repo / "src"))
                p = subprocess.run([py, "-c", CHILD, str(repo / "fixtures" / f"{name}.npz"), str(th), str(n)],
                                   env=env, capture_output=True, text=True)
                kb = [l for l in p.stdout.splitlines() if l.startswith("KB ")]
                row = dict(round=r, model=name, threads=th, flags=fl)
                row.update(json.loads(kb[0][3:]) if kb else dict(error=p.stderr[-400:]))
                open(out, "a").write(json.dumps(row) + "\n")
                print(row.get("model_ms", row.get("error")), name, th, fl, flush=True)
