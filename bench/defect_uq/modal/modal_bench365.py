"""Fits on the re-centred Cantor defect benchmark (bench365): POPS (linear), GP (PCA-128,
host-cache, 4 L-BFGS starts) and tempered ARD (linear, --uq ard), on a LOCAL ace-jax worktree's src
(ACEJAX_SRC; the ard arm needs feat/ard-uq) baked over main.  Outputs: volume acegp-prod-out/bench365_<arm>/ (pred_*.npz, metrics, model).

  modal deploy modal_bench365.py && modal run modal_bench365.py::launch [--arms pops,gp,ard] [--smoke]
"""
import os
import pathlib

import modal

HOME = pathlib.Path.home()
ACEGP = pathlib.Path(os.environ.get("ACEGP_DATA", HOME / "acegp-data"))   # data + generated sets
DATA = ACEGP / "cantor"
_HERE = pathlib.Path(__file__).resolve()
# ace-jax checkout (local only: inside the container this module is /root/modal_bench365.py)
WT = pathlib.Path(os.environ.get("ACEJAX_SRC", _HERE.parents[3] if len(_HERE.parents) > 3 else _HERE.parent))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands("git clone --depth 1 --branch main https://github.com/ACEsuit/ace-jax /root/ace-jax",
                  "pip install -e '/root/ace-jax[gp,cuda]'")
    # the clone layer is cached: keep the core deps of the LOCAL src (ACEJAX_SRC) current
    .pip_install("extxyz>=0.4.5", "jax[cuda12]>=0.10.1", "lineax>=0.1.1", "equinox>=0.11", "scipy>=1.10")
    .add_local_file(str(DATA / "cantor_embed_d16_deg10.npz"), "/data/cantor_embed_d16_deg10.npz")
    .add_local_dir(str(DATA / "bench365"), "/data", ignore=["*.npy", "smoke/out_*/**"])
    .add_local_file(str(pathlib.Path(__file__).parent / "fit_bench.py"), "/root/fit_bench.py")
    .add_local_file(str(pathlib.Path(__file__).parent / "ard_arms.py"), "/root/ard_arms.py")
    .add_local_file(str(pathlib.Path(__file__).parent / "served_arrays.py"), "/root/served_arrays.py")
    .add_local_file(str(pathlib.Path(__file__).parent / "train_extra.py"), "/root/train_extra.py")
    .add_local_dir(str(WT / "src"), "/root/ace-jax/src", ignore=["**/__pycache__/**"])
)
app = modal.App("acegp-bench365")
vol = modal.Volume.from_name("acegp-prod-out")


@app.function(gpu="B200", image=image, volumes={"/out": vol}, timeout=86400, memory=200 * 1024)
def fit_arm(arm: str, smoke: bool = False, tag: str = "", train_extra: str = "") -> int:
    """tag: suffix of the output dir (bench365_<arm><tag>); train_extra: comma-separated container paths of xyz
    files appended to the training set (e.g. /out/defects/big3_cracks_r2-3.xyz)."""
    import subprocess, time
    out = f"/out/bench365_{arm}{tag}{'_smoke' if smoke else ''}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, JAX_ENABLE_X64="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    argv = ["python", "-u", "/root/fit_bench.py", "/data/smoke" if smoke else "/data", out, arm] + \
        (["--smoke"] if smoke else []) + (["--train-extra", train_extra] if train_extra else [])
    t0 = time.time()
    with open(f"{out}/stdout.log", "w") as so, open(f"{out}/stderr.log", "w") as se:
        p = subprocess.Popen(argv, stdout=so, stderr=se, env=env)
        while p.poll() is None:
            time.sleep(120); vol.commit()
    open(f"{out}/rc.txt", "w").write(f"rc={p.returncode} secs={time.time() - t0:.1f}\n")
    vol.commit()
    return p.returncode


@app.local_entrypoint()
def launch(arms: str = "pops,gp", smoke: bool = False, tag: str = "", train_extra: str = ""):
    f = modal.Function.from_name("acegp-bench365", "fit_arm")
    for arm in arms.split(","):
        print("spawned", arm, f.spawn(arm, smoke, tag, train_extra).object_id)


@app.function(gpu="A100-80GB", image=image, volumes={"/out": vol}, timeout=4 * 3600)
def big_errors(run: str = "bench365_pops", xyz: str = "/data/big.xyz", tag: str = "") -> str:
    """Per-atom |F_model - F_MACE| on the big crack / dislocation cells (xyz: data/big.xyz, or e.g.
    /out/defects/big3_mh1.xyz), from the run's saved linear model.  If the run wrote posterior.npz
    (--uq ard), also the per-atom ARD force sigma (ACECalculator(posterior=): node-chunked design
    rows, fine above 2.8k atoms); a schema-3 posterior also serves forces_q, forces_group and forces_cov, saved
    with the error vector dF (validate_shape.py).  Writes <run>/big<tag>_err.npz with the per-atom family,
    r_core, fixed mask and config index, so the file scores without sites.npz."""
    import os
    os.environ["JAX_ENABLE_X64"] = "1"
    import sys
    import time
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    from ase.io import read
    from ace_jax import ACECalculator
    post = f"/out/{run}/posterior.npz"
    calc = ACECalculator(f"/out/{run}/model.npz", posterior=post) if os.path.exists(post) \
        else ACECalculator(f"/out/{run}/model.npz")
    sys.path.insert(0, "/root")
    import served_arrays
    # served properties of a schema-3 posterior; older posteriors serve forces_std only
    props = ["forces_std"]
    if os.path.exists(post):
        if "group_table_json" in np.load(post).files:      # schema 3
            props += ["forces_q", "forces_group", "forces_cov"]
    err, dfv, srv, fam, rc, fx, cid, t0 = [], [], [], [], [], [], [], time.time()
    for k, a in enumerate(read(xyz, ":")):
        fam.append(np.full(len(a), a.info.get("family", "?")))
        rc.append(a.arrays.get("r_core", np.full(len(a), np.nan)))
        fx.append(a.arrays.get("fixed", np.zeros(len(a), bool)).astype(bool))
        cid.append(np.full(len(a), k))
        a.calc = calc
        d = a.get_forces() - a.arrays["mace_force"]
        dfv.append(d.astype(np.float32))
        err.append(np.linalg.norm(d, axis=1))
        if os.path.exists(post):      # E/F reused from the get_forces call above
            srv.append(served_arrays.collect(calc, [a], props))
    out = dict(err=np.concatenate(err), dF=np.concatenate(dfv), family=np.concatenate(fam),
               r_core=np.concatenate(rc), fixed=np.concatenate(fx), cfg=np.concatenate(cid))
    if srv:
        out.update({k: np.concatenate([x[k] for x in srv]) for k in srv[0]})
    sd = out.get("sd", [])
    np.savez(f"/out/{run}/big{tag}_err.npz", **out)
    vol.commit()
    e = out["err"]
    return (f"{len(e)} atoms, median |dF| {np.median(e):.3f}, 99th {np.percentile(e, 99):.3f} eV/A"
            + (f"; median sigma {np.median(out['sd']):.3f}" if len(sd) else "") + f"; {time.time() - t0:.0f} s")


@app.local_entrypoint()
def launch_big(run: str = "bench365_pops", xyz: str = "/data/big.xyz", tag: str = ""):
    print(big_errors.remote(run, xyz, tag))


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


@app.function(gpu="B200", image=image.add_local_file(str(pathlib.Path(__file__).parents[1] / "scoring" / "bayes_joint.py"),
                                                     "/root/bayes_joint.py"),
              volumes={"/out": vol}, timeout=12 * 3600, memory=128 * 1024)
def bayes_joint(n_draws: int = 6, diag: bool = False, opt_only: bool = False) -> int:
    import subprocess
    out = "/out/bench365_bayes_joint"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, JAX_ENABLE_X64="1", XLA_PYTHON_CLIENT_PREALLOCATE="false", **({"DIAG": "1"} if diag else {}), **({"OPT_ONLY": "1", "OPT_LOG": "1"} if opt_only else {}))
    with open(f"{out}/stdout.log", "w") as so, open(f"{out}/stderr.log", "w") as se:
        p = subprocess.Popen(["python", "-u", "/root/bayes_joint.py", "/data", "/out/bench365_pops/theta_map.json",
                              f"{out}/joint.npz", str(n_draws)], stdout=so, stderr=se, env=env)
        while p.poll() is None:
            import time; time.sleep(60); vol.commit()
    open(f"{out}/rc.txt", "w").write(f"rc={p.returncode}\n"); vol.commit()
    return p.returncode


@app.local_entrypoint()
def launch_joint(n_draws: int = 6, diag: bool = False, opt_only: bool = False):
    print("rc", bayes_joint.remote(n_draws, diag, opt_only))


@app.function(gpu="B200", image=image.add_local_file(str(pathlib.Path(__file__).parent / "sandwich_spike.py"),
                                                     "/root/sandwich_spike.py"),
              volumes={"/out": vol}, timeout=6 * 3600, memory=200 * 1024)
def sandwich(run: str = "bench365_ard_v2") -> int:
    """SPIKE: sandwich (misspecification-robust) force variances on an ARD run -> <run>/sandwich.npz."""
    import subprocess
    import time
    out = f"/out/{run}"
    env = dict(os.environ, JAX_ENABLE_X64="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    with open(f"{out}/sandwich.log", "w") as so:
        p = subprocess.Popen(["python", "-u", "/root/sandwich_spike.py", "/data", out], stdout=so,
                             stderr=subprocess.STDOUT, env=env)
        while p.poll() is None:
            time.sleep(60); vol.commit()
    vol.commit()
    return p.returncode


@app.local_entrypoint()
def launch_sandwich(run: str = "bench365_ard_v2"):
    print("spawned sandwich", sandwich.spawn(run).object_id)
