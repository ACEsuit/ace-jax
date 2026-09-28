"""Fits on the re-centred Cantor defect benchmark (bench365): POPS (linear) and GP (PCA-128,
host-cache, 4 L-BFGS starts), on the LOCAL ace-jax feat/pops-perf worktree (src baked over the
pushed branch).  Outputs: volume acegp-prod-out/bench365_<arm>/ (pred_*.npz, metrics, model).

  modal deploy modal_bench365.py && modal run modal_bench365.py::launch [--arms pops,gp] [--smoke]
"""
import os
import pathlib

import modal

HOME = pathlib.Path.home()
ACEGP = pathlib.Path(os.environ.get("ACEGP_DATA", HOME / "acegp-data"))   # data + generated sets
DATA = HOME / "acegp-data" / "cantor"
WT = pathlib.Path(os.environ.get("ACEJAX_SRC", pathlib.Path(__file__).resolve().parents[3]))   # ace-jax checkout
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands("git clone --depth 1 --branch feat/pops-perf https://github.com/ACEsuit/ace-jax /root/ace-jax",
                  "pip install -e '/root/ace-jax[gp,cuda]'")
    .add_local_file(str(DATA / "cantor_embed_d16_deg10.npz"), "/data/cantor_embed_d16_deg10.npz")
    .add_local_dir(str(DATA / "bench365"), "/data", ignore=["*.npy", "smoke/out_*/**"])
    .add_local_file(str(pathlib.Path(__file__).parent / "fit_bench.py"), "/root/fit_bench.py")
    .add_local_dir(str(WT / "src"), "/root/ace-jax/src", ignore=["**/__pycache__/**"])
)
app = modal.App("acegp-bench365")
vol = modal.Volume.from_name("acegp-prod-out")


@app.function(gpu="B200", image=image, volumes={"/out": vol}, timeout=86400, memory=200 * 1024)
def fit_arm(arm: str, smoke: bool = False) -> int:
    import subprocess, time
    out = f"/out/bench365_{arm}{'_smoke' if smoke else ''}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, JAX_ENABLE_X64="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    argv = ["python", "-u", "/root/fit_bench.py", "/data/smoke" if smoke else "/data", out, arm] + \
        (["--smoke"] if smoke else [])
    t0 = time.time()
    with open(f"{out}/stdout.log", "w") as so, open(f"{out}/stderr.log", "w") as se:
        p = subprocess.Popen(argv, stdout=so, stderr=se, env=env)
        while p.poll() is None:
            time.sleep(120); vol.commit()
    open(f"{out}/rc.txt", "w").write(f"rc={p.returncode} secs={time.time() - t0:.1f}\n")
    vol.commit()
    return p.returncode


@app.local_entrypoint()
def launch(arms: str = "pops,gp", smoke: bool = False):
    f = modal.Function.from_name("acegp-bench365", "fit_arm")
    for arm in arms.split(","):
        print("spawned", arm, f.spawn(arm, smoke).object_id)


@app.function(gpu="A100-80GB", image=image, volumes={"/out": vol}, timeout=4 * 3600)
def big_errors(run: str = "bench365_pops") -> str:
    """Per-atom |F_model - F_MACE| on the big crack / dislocation cells (data/big.xyz), from the
    run's saved linear model (ACECalculator: autodiff forces, no design rows -- the design-row
    path does not fit one >2.8k-atom cell in int32-indexed GEMMs)."""
    import os
    os.environ["JAX_ENABLE_X64"] = "1"
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    from ase.io import read
    from ace_jax import ACECalculator
    calc = ACECalculator(f"/out/{run}/model.npz")
    err = []
    for a in read("/data/big.xyz", ":"):
        a.calc = calc
        err.append(np.linalg.norm(a.get_forces() - a.arrays["mace_force"], axis=1))
    np.savez(f"/out/{run}/big_err.npz", err=np.concatenate(err))
    vol.commit()
    e = np.concatenate(err)
    return f"{len(e)} atoms, median |dF| {np.median(e):.3f}, 99th {np.percentile(e, 99):.3f} eV/A"


@app.local_entrypoint()
def launch_big(run: str = "bench365_pops"):
    print(big_errors.remote(run))


@app.function(gpu="B200", image=image.add_local_file(str(pathlib.Path(__file__).parents[1] / "scoring" / "bayes_bodyorder.py"),
                                                     "/root/bayes_bodyorder.py"),
              volumes={"/out": vol}, timeout=12 * 3600, memory=128 * 1024)
def bayes_bodyorder() -> int:
    import subprocess
    out = "/out/bench365_bayes"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, JAX_ENABLE_X64="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    with open(f"{out}/stdout.log", "w") as so, open(f"{out}/stderr.log", "w") as se:
        p = subprocess.Popen(["python", "-u", "/root/bayes_bodyorder.py", "/data",
                              "/out/bench365_pops/theta_map.json", f"{out}/bayes.npz"], stdout=so, stderr=se, env=env)
        while p.poll() is None:
            import time; time.sleep(60); vol.commit()
    open(f"{out}/rc.txt", "w").write(f"rc={p.returncode}\n"); vol.commit()
    return p.returncode


@app.local_entrypoint()
def launch_bayes():
    print("rc", bayes_bodyorder.remote())
