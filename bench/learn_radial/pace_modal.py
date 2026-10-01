"""Run pacemaker (python-ace + TensorPotential) fits on Modal GPUs, in parallel,
each followed by the validation evaluation used for the learned-radial comparison.

The image pins the stack used on lestrade: tensorflow[and-cuda] 2.16.2,
numpy<=1.26.4 (TensorPotential's pin), TensorPotential @1e44b25, python-ace
@66c35ea. The local directory PACE_DATA holds the datasets, their meta json,
eval_pace.py, the reference xyz, pace_requirements.txt (frozen from the lestrade env) and one
input_<name>.yaml per fit. Each input
refers to ../<system>_train.pckl.gzip, so a fit runs in /work/<name> with the
data in /work.

    PACE_DATA=/path/to/pace_modal modal run bench/learn_radial/pace_modal.py \
        --names sige_fs,sige_k0.02,sige_k0.1 --xyz sige_mh1.xyz --meta sige_meta.json \
        --ntrain 200 --out runs/pace_modal
"""
import os
import pathlib
import shutil
import subprocess

import modal

DATA = pathlib.Path(os.environ.get("PACE_DATA", "/data")) if modal.is_local() else pathlib.Path("/data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "g++")
    # the exact package set of the working lestrade env (uv pip freeze), then the two
    # ICAMS packages without dependency resolution (TensorPotential declares python<3.11
    # but works on 3.11, as on lestrade)
    .pip_install_from_requirements(str(DATA / "pace_requirements.txt") if modal.is_local() else "/dev/null")
    .run_commands(
        "git clone -q https://github.com/ICAMS/TensorPotential.git /src/TensorPotential",
        "cd /src/TensorPotential && git checkout -q 1e44b25 && pip install -q --no-deps --ignore-requires-python .",
        "git clone -q https://github.com/ICAMS/python-ace.git /src/python-ace",
        "cd /src/python-ace && git checkout -q 66c35ea && pip install -q --no-deps .",
    )
    .add_local_dir(DATA, "/data")
)
app = modal.App("ace-jax-pacemaker", image=image)


@app.function(gpu="A100-80GB", timeout=10 * 3600)
def fit(name: str, xyz: str, meta: str, ntrain: int) -> dict:
    work = pathlib.Path("/work")
    run = work / name
    run.mkdir(parents=True, exist_ok=True)
    for p in pathlib.Path("/data").iterdir():
        if p.name.endswith(".pckl.gzip"):
            shutil.copy(p, work / p.name)
    shutil.copy(f"/data/input_{name}.yaml", run / "input.yaml")
    r = subprocess.run(["pacemaker", "input.yaml"], cwd=run, capture_output=True, text=True)
    log = r.stdout + r.stderr
    out = {"returncode": r.returncode, "log_tail": "\n".join(log.splitlines()[-40:])}
    pot = run / "output_potential.yaml"
    if pot.exists():
        e = subprocess.run(["python", "/data/eval_pace.py", str(pot), f"/data/{xyz}", f"/data/{meta}", str(ntrain)],
                           capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        out["eval"] = (e.stdout + e.stderr).strip().splitlines()[-1] if (e.stdout + e.stderr).strip() else ""
        out["potential"] = pot.read_text()
    out["fit_log"] = log
    return out


@app.local_entrypoint()
def main(names: str, xyz: str, meta: str, ntrain: int = 200, out: str = "runs/pace_modal"):
    calls = [(n, xyz, meta, ntrain) for n in names.split(",")]
    for (n, *_), res in zip(calls, fit.starmap(calls)):
        d = pathlib.Path(out) / n
        d.mkdir(parents=True, exist_ok=True)
        (d / "fit.log").write_text(res["fit_log"])
        if "potential" in res:
            (d / "output_potential.yaml").write_text(res["potential"])
        print(f"== {n} rc={res['returncode']}\n{res.get('eval', res['log_tail'][-2000:])}", flush=True)
