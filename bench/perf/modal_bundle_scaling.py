"""bench/perf/bundle_scaling.py on a Modal A100-80GB (the benchmark image).

    uv run --with modal modal run bench/perf/modal_bundle_scaling.py::main \
        --cases round1|round2|trace|trace_ace --tag <tag>
    uv run --with modal modal run bench/perf/modal_bundle_scaling.py::lammps \
        --model pace_Cantor_medium --ns 16384,131072 --variants "stock,stock@65536" --tag <tag>

`main`: every (model, n) case in its own process (one container each, in
parallel), its variants in sequence; rows land in
bench/perf/results/bundle_scaling_<tag>.json.  `lammps`: one LAMMPS run per
(n, variant) with a bundle built by bench/perf/lammps_bundle_variant.py, for
the end-to-end check.
"""
import json
import pathlib
import sys

import modal

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scaling.modal_app import base_image, with_sources  # noqa: E402

app = modal.App("ace-jax-bundle-scaling", image=with_sources(base_image))
OUT = HERE / "results"
ENV = {"PYTHONPATH": "/ace-jax/bench:/ace-jax/src", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}

ALL = "stock,chunked,k_tight,pack,model_fb"
CASES = ([("pace_Cantor_medium", n, ALL) for n in (4096, 16384, 65536, 131072, 262144)]
         + [("ace_Cantor_medium", n, ALL) for n in (4096, 16384, 32768)]
         + [("ace_Cantor_medium", n, ALL + ",sparse") for n in (65536, 131072)])

# round 2: chunk size, checkpointing, tight slots
PACE2 = "model_fb@32768,model_fb@65536,stock@32768,stock@65536,stock@65536!,k_tight@65536"
ACE2 = "model_fb@16384,model_fb@32768,stock@32768,stock@65536,stock@32768!,k_tight@32768,k_tight@65536"
ROUND2 = ([("pace_Cantor_medium", n, PACE2) for n in (16384, 65536, 131072, 262144)]
          + [("pace_Cantor_medium", 524288, "stock@65536,k_tight@65536")]
          + [("ace_Cantor_medium", n, ACE2) for n in (16384, 65536, 131072)])
TRACE = [("pace_Cantor_medium", n, v) for n in (65536, 262144) for v in ("model_fb", "model_fb@65536")]
TRACE_ACE = [("ace_Cantor_medium", 65536, v) for v in ("model_fb", "model_fb@32768")]
SETS = {"round1": CASES, "round2": ROUND2, "trace": TRACE, "trace_ace": TRACE_ACE}


@app.function(gpu="A100-80GB", timeout=3600)
def case(model: str, n: int, variants: str, order: str = "interleaved", trace: bool = False):
    import os
    import subprocess
    import time
    t0 = time.time()
    env = {**os.environ, **ENV}
    if trace:       # one HLO dump per variant would be ambiguous: trace one variant per case
        env.update(ACEJAX_DUMP="/tmp/dump", XLA_FLAGS="--xla_dump_to=/tmp/dump --xla_dump_hlo_as_text")
    p = subprocess.run([sys.executable, "/ace-jax/bench/perf/bundle_scaling.py", model, str(n),
                        "--variants", variants, "--order", order, "--budget", "3"]
                       + (["--trace"] if trace else []),
                       capture_output=True, text=True, env=env, timeout=3500)
    rows = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    lj = subprocess.run("git -C /opt/lammps-jax log -1 --format='%h %cs %s'", shell=True,
                        capture_output=True, text=True).stdout.strip()
    for r in rows:
        r["lammps_jax"] = lj
        r["case_wall_s"] = time.time() - t0
    if p.returncode or not rows:
        rows.append({"model": model, "n": n, "status": "crash", "rc": p.returncode,
                     "stderr": p.stderr[-3000:]})
    return rows


@app.local_entrypoint()
def main(tag: str = "scaling", only: str = "", order: str = "interleaved", cases: str = "round1"):
    trace = cases.startswith("trace")
    cases = [c for c in SETS[cases] if not only or c[0].startswith(only)]
    rows = []
    for out in case.starmap([(m, n, v, order, trace) for m, n, v in cases]):
        rows += out
        for r in out:
            print(r.get("model"), r.get("n"), r.get("variant"), r.get("status"),
                  r.get("median_s"), (r.get("mem") or {}).get("temp_size_in_bytes"))
    OUT.mkdir(exist_ok=True)
    path = OUT / f"bundle_scaling_{tag}.json"
    path.write_text(json.dumps(rows, indent=1) + "\n")
    print("->", path)


@app.function(gpu="A100-80GB", timeout=3600)
def lammps_case(model: str, n: int, variant: str, steps: int = 50, warmup: int = 10):
    import os
    import subprocess
    # export on the CPU: the exporting process then holds no GPU pool while LAMMPS runs
    p = subprocess.run([sys.executable, "/ace-jax/bench/perf/lammps_bundle_variant.py", model,
                        str(n), "--variant", variant, "--steps", str(steps), "--warmup",
                        str(warmup), "--work", f"/tmp/lbv_{variant}"],
                       capture_output=True, text=True, timeout=3500,
                       env={**os.environ, **ENV, "JAX_PLATFORMS": "cpu"})
    rows = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    return rows[-1] if rows else {"model": model, "n": n, "variant": variant, "status": "crash",
                                  "stderr": p.stderr[-3000:]}


@app.local_entrypoint()
def lammps(model: str = "pace_Cantor_medium", ns: str = "16384,131072",
           variants: str = "stock,chunked", tag: str = "lammps", steps: int = 50):
    jobs = [(model, int(n), v, steps) for n in ns.split(",") for v in variants.split(",")]
    rows = list(lammps_case.starmap(jobs))
    for r in rows:
        print(r.get("model", model), r.get("n_atoms", r.get("n")), r.get("variant"),
              r.get("status"), r.get("step_s"), r.get("atom_steps_per_s"), r.get("pe"),
              r.get("error", "")[:300] if r.get("status") != "ok" else "")
    OUT.mkdir(exist_ok=True)
    path = OUT / f"bundle_scaling_{tag}.json"
    path.write_text(json.dumps(rows, indent=1) + "\n")
    print("->", path)
