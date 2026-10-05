"""Step 5 of docs/dev/force-uq-vs-calm.md on GPUs: the served shape by the design rows vs the r-output linear ACE
(ACECalculator(shape_path="committee")) and rank truncation, on the v3 big cells (scoring/shape_eval.py).

A100-40GB is the target the plan names (the rows path needs A100-80GB-class memory for 3-4k-atom cells, per
the how-to); A100-80GB gives the rows path's timing.  Model and posterior: volume acegp-prod-out/<run>/.
Writes /out/<run>/shape_eval_<gpu>.md (and the per-variant npz beside it) and returns the table.

  modal run modal_shape_eval.py::launch [--run bench365_ard_default_pol] [--gpus a100_40,a100_80] [--cells 3]
"""
import os
import pathlib

import modal

_HERE = pathlib.Path(__file__).resolve()
WT = pathlib.Path(os.environ.get("ACEJAX_SRC", _HERE.parents[3] if len(_HERE.parents) > 3 else _HERE.parent))
SCORING = _HERE.parents[1] / "scoring" if len(_HERE.parents) > 1 else _HERE.parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands("git clone --depth 1 --branch main https://github.com/ACEsuit/ace-jax /root/ace-jax",
                  "pip install -e '/root/ace-jax[cuda]'")
    .pip_install("extxyz>=0.4.5", "jax[cuda12]>=0.10.1", "lineax>=0.1.1", "equinox>=0.11", "scipy>=1.10")
    .add_local_file(str(SCORING / "shape_eval.py"), "/root/scoring/shape_eval.py")
    .add_local_file(str(SCORING / "_frames.py"), "/root/scoring/_frames.py")
    .add_local_dir(str(WT / "src"), "/root/ace-jax/src", ignore=["**/__pycache__/**"])
)
app = modal.App("acegp-shape-eval")
vol = modal.Volume.from_name("acegp-prod-out")
VARIANTS = "rows,committee,committee:800,committee:400,committee:200,committee:100,committee:50"


def _run(tag, run, cells, variants):
    import subprocess
    d = f"/out/{run}"
    env = dict(os.environ, JAX_ENABLE_X64="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    out = f"{d}/shape_eval_{tag}.md"
    r = subprocess.run(["python", "-u", "/root/scoring/shape_eval.py", "--model", f"{d}/model.npz", "--posterior",
                        f"{d}/posterior.npz", "--cells", f"/out/defects/big3_mh1.xyz:{cells}", "--force-key",
                        "mace_force", "--variants", variants, "--out", out], capture_output=True, text=True, env=env)
    vol.commit()
    return f"### {tag}\n\n{r.stdout[-8000:]}\n" + (f"\nstderr tail:\n{r.stderr[-3000:]}" if r.returncode else "")


@app.function(gpu="A100", image=image, volumes={"/out": vol}, timeout=6 * 3600, memory=96 * 1024)
def shape_eval_a100_40(run: str, cells: int, variants: str) -> str:
    return _run("a100_40", run, cells, variants)


@app.function(gpu="A100-80GB", image=image, volumes={"/out": vol}, timeout=6 * 3600, memory=128 * 1024)
def shape_eval_a100_80(run: str, cells: int, variants: str) -> str:
    return _run("a100_80", run, cells, variants)


@app.local_entrypoint()
def launch(run: str = "bench365_ard_default_pol", gpus: str = "a100_40,a100_80", cells: int = 3,
           variants: str = VARIANTS):
    fns = {"a100_40": shape_eval_a100_40, "a100_80": shape_eval_a100_80}
    calls = [fns[g].spawn(run, cells, variants) for g in gpus.split(",")]
    for c in calls:
        print(c.get())
