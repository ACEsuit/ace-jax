"""ACE vs PACE gap runs on a Modal A100 (docs/ace-vs-pace-gap.md), every case on
ONE container so the models compare on one card.

    uv run --with modal modal run bench/perf/modal_gap.py::profile   # §2 stage profiles
    uv run --with modal modal run bench/perf/modal_gap.py::bench     # §4 interleaved A/B
    uv run --with modal modal run bench/perf/modal_gap.py::learned   # learned (analytic) radials

Results land in bench/perf/results/gap_*.json (nothing in bench/scaling/results).
"""
import json
import pathlib
import sys

import modal

HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), "/ace-jax/bench/perf", "/ace-jax/bench"):
    sys.path.insert(0, _p)                  # locally, and in the container (/root/modal_gap.py)
from modal_profile import image  # noqa: E402

app = modal.App("ace-jax-gap", image=image)
vol = modal.Volume.from_name("ace-jax-perf-profile", create_if_missing=True)
OUT = HERE / "results"
M = "/ace-jax/bench/scaling/models"


@app.function(gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def run(argvs: list, tag: str, timeout: int = 1500):
    import os
    import subprocess
    import tempfile
    gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True,
                         capture_output=True, text=True).stdout.strip()
    out = []
    for argv in argvs:
        argv = list(argv)
        if "--trace" in argv:                           # fresh trace / dump dirs per case
            argv[argv.index("--trace") + 1] = tempfile.mkdtemp()
            argv[argv.index("--dump") + 1] = tempfile.mkdtemp()
        p = subprocess.run(["python"] + argv, capture_output=True, text=True, cwd="/ace-jax",
                           env={**os.environ, "PYTHONPATH": "/ace-jax/bench:/ace-jax/src"},
                           timeout=timeout)
        try:
            r = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:                                              # noqa: BLE001
            r = {"argv": argv, "error": p.stdout[-1500:] + p.stderr[-3000:]}
        r["gpu"] = gpu
        r["argv"] = argv
        out.append(r)
        pathlib.Path(f"/vol/gap{tag}.json").write_text(json.dumps(out))
        vol.commit()
    return out


def _save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print("->", OUT / name)


@app.local_entrypoint()
def profile(models: str = "Cantor_medium,SiGe_medium", n: int = 8192,
            fast: str = "compact+pairfold", tag: str = ""):
    argvs = []
    for m in models.split(","):
        s = m.split("_")[0]
        tr = ["--trace", "T", "--dump", "D"]
        argvs += [["bench/perf/profile_gap.py", f"{M}/ace_{m}.npz", s, str(n)] + tr,
                  ["bench/perf/profile_gap.py", f"{M}/pace_{m}.yace", s, str(n)] + tr]
        if fast:
            argvs.append(["bench/perf/profile_gap.py", f"{M}/ace_{m}.npz", s, str(n),
                          "--variant", fast] + tr)
    res = run.remote(argvs, "_profile" + tag)
    _save(f"gap_profile_{n}{tag}.json", res)
    for r in res:
        k = r.get("kernels", {})
        print(r.get("path"), r.get("variant"), r.get("efv_s"), r.get("error", "")[:300],
              {s_: v["us"] for s_, v in k.get("stages", {}).items()})


@app.local_entrypoint()
def bench(models: str = "Cantor_medium,SiGe_medium", ns: str = "1024,8192,32768,131072",
          variants: str = "prune,prune+lblock,compact,compact+pairfold",
          extra: str = "Cantor_small,Cantor_large,SiGe_small,SiGe_large", calc_n: int = 8192,
          extra_variants: str = "compact+pairfold", pace_variants: str = "", extra_n: int = 8192,
          rounds: int = 3, tag: str = ""):
    argvs = []
    for m in models.split(","):
        for n in ns.split(","):
            argvs.append(["bench/perf/ace_gap_bench.py", m, n, "--variants", variants,
                          "--pace-variants", pace_variants, "--rounds", str(rounds)]
                         + (["--calc"] if int(n) == calc_n else []))
    for m in [x for x in extra.split(",") if x]:
        argvs.append(["bench/perf/ace_gap_bench.py", m, str(extra_n), "--variants",
                      extra_variants, "--pace-variants", pace_variants, "--rounds", str(rounds)])
    res = run.remote(argvs, "_bench" + tag)
    _save(f"gap_bench{tag}.json", res)
    for r in res:
        print(r.get("model"), r.get("n"), r.get("gpu"), r.get("error", "")[:300],
              {k: (round(c["efv_s"] * 1e3, 2), round(c["vs_ace"], 2),
                   round(c.get("calc_s", 0) * 1e3, 2)) for k, c in r.get("cases", {}).items()})


@app.local_entrypoint()
def learned(models: str = "SiGe_medium,Cantor_medium", n: int = 8192, nqs: str = "8,12,16,20",
            filled_nq: int = 12, rounds: int = 3, reps: int = 10,
            extra: str = "SiGe_large,Cantor_large", spline_tol: float = 1e-10, tag: str = ""):
    """Learned (analytic) radials vs splines and the lean form of each, with
    to_spline (bench/perf/learned_radial_bench.py; docs/learned-radial-splining.md),
    every model on ONE container."""
    argvs = [["bench/perf/learned_radial_bench.py", m, str(n), "--nqs", nqs, "--filled-nq",
              str(filled_nq), "--rounds", str(rounds), "--reps", str(reps), "--oldgather",
              "--spline-tol", str(spline_tol)]
             for m in models.split(",")]
    argvs += [["bench/perf/learned_radial_bench.py", m, str(n), "--nqs", "12", "--filled-nq", "0",
               "--rounds", str(rounds), "--reps", str(reps), "--spline-tol", str(spline_tol)]
              for m in extra.split(",") if m]
    res = run.remote(argvs, "_learned" + tag, 900)
    _save(f"learned_radial_{n}{tag}.json", res)
    for r in res:
        print(r.get("model"), r.get("gpu"), r.get("error", "")[:300], r.get("info"))
        for k, c in r.get("cases", {}).items():
            print(f"  {k:28s} {c['efv_s'] * 1e3:8.2f} ms  x{c['vs_spline']:.2f}")
