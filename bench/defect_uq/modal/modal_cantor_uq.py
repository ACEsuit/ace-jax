"""Cantor production UQ runs on Modal (B200), from ace-jax PR #8 (feat/multistart-map).

  1. cantor_gp_pca128_ms4  joint ACE+GP, PCA-128 residual features, host-cached LML,
                           best of 4 L-BFGS MAP starts (the joint LML is multimodal)
  2. cantor_linear_pops    linear ACE + POPS misspecification UQ (BLR mean,
                           per-quantity ridges chosen by CRPS on a train hold-out)

Both: production basis cantor_embed_d16_deg10 (L = 27.6k), cantor4k split
3200 train / 800 test (seed-0 permutation), OOD = cantor_ood2_mh1 (200 compress +
200 single-vacancy configs whose parents are test configs perm[3200:3400]).

  modal deploy modal_cantor_uq.py && modal run modal_cantor_uq.py::launch
  outputs: volume acegp-prod-out/<name>/ (rc.txt, stdout.log, metrics.json, pred_*.npz, ...)
"""
import os
import pathlib

import modal

DATA = pathlib.Path(os.environ.get("ACEGP_DATA", pathlib.Path.home() / "acegp-data")) / "cantor"
BRANCH = "main"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        f"git clone --depth 1 --branch {BRANCH} https://github.com/ACEsuit/ace-jax /root/ace-jax",
        "pip install -e '/root/ace-jax[gp,cuda]'",
    )
    .add_local_file(str(DATA / "cantor_embed_d16_deg10.npz"), "/data/cantor_embed_d16_deg10.npz")
    .add_local_file(str(DATA / "cantor4k_b_mh1.xyz"), "/data/cantor4k_b_mh1.xyz")
    .add_local_file(str(DATA / "cantor_ood2_mh1.xyz"), "/data/cantor_ood2_mh1.xyz")
)

app = modal.App("acegp-cantor-uq")
vol = modal.Volume.from_name("acegp-prod-out", create_if_missing=True)

KEYS = ["--energy-key", "mace_energy", "--force-key", "mace_force", "--virial-key", "mace_virial"]
CANTOR = ["--model", "/data/cantor_embed_d16_deg10.npz", "--data", "/data/cantor4k_b_mh1.xyz",
          "--ood", "/data/cantor_ood2_mh1.xyz", "--r0", "2.5", "--ntrain", "3200",
          "--test-start", "3200", "--ntest", "800", "--batch", "4", "--no-predict-train",
          "--rungs", "map", "--map-steps", "40", *KEYS]

JOBS = {
    "cantor_gp_pca128_ms4": [*CANTOR, "--arm", "gp", "--m-per-species", "100",
                             "--density", "pca", "--pca-d", "128", "--lml", "host-cache",
                             "--opt", "lbfgs", "--map-restarts", "4"],
    "cantor_linear_pops":   [*CANTOR, "--arm", "linear", "--uq", "pops", "--pops-ridge", "auto"],
}
# 2026-09-27: 3 of 4 starts hit the 40-iteration cap -- rerun to convergence
JOBS["cantor_gp_pca128_ms4_s100"] = [*JOBS["cantor_gp_pca128_ms4"], "--map-steps", "100"]


@app.function(gpu="B200", image=image, volumes={"/out": vol}, timeout=86400,
              memory=200 * 1024)   # MiB: host-cached linear rows (~106 GB at Cantor-3200)
def run_one(name: str, argv: list[str]) -> dict:
    import hashlib, subprocess, time
    out = f"/out/{name}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, JAX_ENABLE_X64="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    rev = subprocess.run(["git", "-C", "/root/ace-jax", "rev-parse", "HEAD"], capture_output=True, text=True)
    open(f"{out}/commit.txt", "w").write(rev.stdout)
    open(f"{out}/run_py.md5", "w").write(
        hashlib.md5(open("/root/ace-jax/bench/acegp_cantor/run.py", "rb").read()).hexdigest())
    mon = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                            "-l", "5"], stdout=open(f"{out}/gpu_mem_mib.log", "w"))
    t0 = time.time()
    with open(f"{out}/stdout.log", "w") as so, open(f"{out}/stderr.log", "w") as se:
        # stream to the volume so progress is visible while it runs
        p = subprocess.Popen(["python", "-u", "/root/ace-jax/bench/acegp_cantor/run.py", *argv, "--out", out],
                             stdout=so, stderr=se, env=env)
        while p.poll() is None:
            time.sleep(120); vol.commit()
    dt = time.time() - t0
    mon.terminate()
    mem = [int(x) for x in open(f"{out}/gpu_mem_mib.log").read().split() if x.isdigit()]
    open(f"{out}/rc.txt", "w").write(f"rc={p.returncode} secs={dt:.1f} peak_gpu_mib={max(mem, default=0)}\n")
    vol.commit()
    return {"name": name, "rc": p.returncode, "secs": round(dt, 1)}


@app.local_entrypoint()
def launch():
    f = modal.Function.from_name("acegp-cantor-uq", "run_one")
    for name, argv in JOBS.items():
        print(f"spawned {name} -> {f.spawn(name, argv).object_id}")


@app.local_entrypoint()
def launch_one(name: str):
    f = modal.Function.from_name("acegp-cantor-uq", "run_one")
    print(f"spawned {name} -> {f.spawn(name, JOBS[name]).object_id}")
