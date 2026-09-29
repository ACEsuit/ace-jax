"""bench/perf/lammps_matrix_ab.py on a Modal A100-80GB, one container, interleaved.

    uv run --with modal modal run bench/perf/modal_matrix_ab.py --tag matrix_ab

In one container: a `run 0` parity check of the matrix bundles (list-width and
tight model slots) against the ace-jax calculator at 256 atoms, then per
(model, n) case the three bundles (stock, matrix, tight; exported on the CPU
in a child process so no JAX pool holds the GPU while LAMMPS runs) timed in
interleaved order: stock and matrix on the benchmark deck, all three on the
static stable-MD proxy deck.  Rows land in
bench/perf/results/matrix_ab_<tag>.json.
"""
import json
import pathlib
import sys

import modal

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scaling.modal_app import base_image, with_sources  # noqa: E402

app = modal.App("ace-jax-matrix-ab", image=with_sources(base_image))
OUT = HERE / "results"
ENV = {"PYTHONPATH": "/ace-jax/bench:/ace-jax/src", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
MODELS = ("pace_Cantor_medium", "ace_Cantor_medium", "ace_SiGe_medium")
NS = (16384, 131072)
STEPS = {16384: (200, 20), 131072: (60, 10)}          # timed steps, warm-up


def _sh(cmd, env=None, timeout=3000):
    import os
    import subprocess
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, **ENV, **(env or {})})
    rows = [json.loads(ln) for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode or not rows:
        return {"status": "crash", "rc": p.returncode, "stderr": p.stderr[-3000:]}
    return rows[-1]


def _export(model, n, variant, out, ref=False):
    return _sh([sys.executable, "/ace-jax/bench/perf/lammps_matrix_ab.py", "export", model,
                str(n), variant, out] + (["--ref"] if ref else []), env={"JAX_PLATFORMS": "cpu"})


@app.function(gpu="A100-80GB", timeout=3300)
def ab(models: str, ns: str, parity: bool = True, rounds: int = 1):
    import subprocess
    import time
    sys.path.insert(0, "/ace-jax/bench/perf")
    sys.path.insert(0, "/ace-jax/bench")
    import lammps_matrix_ab as m
    t0 = time.time()
    info = {"device_name": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                          capture_output=True, text=True).stdout.strip(),
            "lammps_jax": subprocess.run("git -C /opt/lammps-jax log -1 --format='%h %cs'",
                                         shell=True, capture_output=True, text=True).stdout.strip()}
    rows = []

    def emit(r):
        r.update(info)
        r["t"] = round(time.time() - t0, 1)
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("ref_F", "stderr")}), flush=True)

    if parity:
        for model in models.split(","):
            for variant in ("matrix", "tight"):
                b = f"/tmp/par_{model}_{variant}.json"
                ex = _export(model, 256, variant, b, ref=True)
                if ex.get("status") == "crash":
                    emit({"kind": "parity", "model": model, "variant": variant, **ex})
                    continue
                r = m.run(model, 256, b, f"/tmp/par_{model}_{variant}", parity=(ex["ref_E"], ex["ref_F"]))
                ex.pop("ref_F")
                emit({"kind": "parity", **ex, **r})
    for model in models.split(","):
        for n in (int(x) for x in ns.split(",")):
            steps, warmup = STEPS.get(n, (60, 10))
            bundles = {}
            for variant in ("stock", "matrix", "tight"):
                b = f"/tmp/ab_{model}_{n}_{variant}.json"
                ex = _export(model, n, variant, b)
                if ex.get("status") == "crash":
                    emit({"kind": "export", "model": model, "n": n, "variant": variant, **ex})
                    continue
                bundles[variant] = (b, ex)
            plan = [("bench", "stock"), ("bench", "matrix"), ("static", "stock"),
                    ("static", "matrix"), ("static", "tight")]
            for rnd in range(rounds):
                for deck, variant in (plan if rnd % 2 == 0 else plan[::-1]):
                    if variant not in bundles:
                        continue
                    b, ex = bundles[variant]
                    r = m.run(model, n, b, f"/tmp/w_{model}_{n}_{deck}_{variant}", deck,
                              steps, warmup, repeats=2)
                    emit({"kind": "time", "variant": variant, "round": rnd, **ex, **r})
    return rows


@app.local_entrypoint()
def main(tag: str = "matrix_ab", models: str = ",".join(MODELS), ns: str = "16384,131072",
         parity: bool = True, rounds: int = 1):
    rows = ab.remote(models, ns, parity, rounds)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{tag}.json"
    path.write_text(json.dumps(rows, indent=1) + "\n")
    for r in rows:
        print(r.get("kind"), r.get("model"), r.get("n"), r.get("deck"), r.get("variant"),
              r.get("status"), r.get("step_s") or (r.get("dE_per_atom"), r.get("max_dF")),
              r.get("pe0"), r.get("widest_row"), r.get("log_errors"))
    print("->", path)
