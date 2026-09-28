"""Run bench/learn_radial/run.py (and rmse.py on its candidates) on Modal GPUs,
several step budgets (and learn_sigma_e_mult values, --mults) in parallel.

ace-jax `src/` and `bench/learn_radial/` are mounted from this checkout, so code
changes need no image rebuild. The model/xyz files come from a local directory
given by LEARN_RADIAL_DATA (sige_base.npz, sige_mh1.xyz, cantor_d4.npz,
cantor1k_b_mh1.xyz). Each run's outputs (summary, radial info, learned radials
per lambda, rmse.json, log) are written under --out/<system>_s<steps>/.

    LEARN_RADIAL_DATA=/path/to/data modal run bench/learn_radial/modal_run.py \
        --system sige --steps 100,200,400 --lam-grid 0.1 --out runs/modal
"""
import os
import pathlib
import subprocess

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/ace-jax")
DATA = pathlib.Path(os.environ.get("LEARN_RADIAL_DATA", "/data")) if modal.is_local() else pathlib.Path("/data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("jax[cuda12]==0.10.2", "equinox>=0.11", "numpy<2.3", "scipy", "lineax>=0.1.1",
                 "ase>=3.22", "pyyaml", "numpyro>=0.16", "optax>=0.2", "blackjax>=1.6")
    .env({"PYTHONPATH": "/ace-jax/src", "JAX_ENABLE_X64": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
    .add_local_dir(ROOT / "src", "/ace-jax/src")
    .add_local_dir(ROOT / "bench" / "learn_radial", "/ace-jax/bench/learn_radial")
    .add_local_dir(DATA, "/data")
)
app = modal.App("ace-jax-learn-radial", image=image)

SYSTEMS = {
    "sige": ["--model", "/data/sige_base.npz", "--data", "/data/sige_mh1.xyz", "--r0", "2.35",
             "--ntrain", "200", "--nval", "100"],
    "cantor": ["--model", "/data/cantor_d4.npz", "--data", "/data/cantor1k_b_mh1.xyz", "--r0", "2.5",
               "--ntrain", "150", "--nval", "100"],
}
KEYS = ["--energy-key", "mace_energy", "--force-key", "mace_force", "--virial-key", "mace_virial"]


def _drop(args, *flags):
    """args with each --flag/value pair removed (rmse_npz.py has no --r0/--n-q)."""
    args = list(args)
    for f in flags:
        if f in args:
            i = args.index(f)
            del args[i:i + 2]
    return args


@app.function(gpu="A100-80GB", timeout=8 * 3600)
def learn(system: str, steps: int, lam_grid: str, reprofile_every: int, n_q: int, mult: float = 1.0,
         density: str = "none", P: int = 1, density_mode: str = "joint", lam_eta_grid: str = "0") -> dict:
    out = pathlib.Path(f"/tmp/out_{system}_s{steps}_m{mult:g}_{density}{P}")
    args = SYSTEMS[system] + KEYS + ["--n-q", str(n_q)]
    r = subprocess.run(["python", "/ace-jax/bench/learn_radial/run.py", *args, "--steps", str(steps),
                        "--reprofile-every", str(reprofile_every), "--lam-grid", lam_grid,
                        "--learn-sigma-e-mult", str(mult), "--density", density, "--P", str(P),
                        "--density-mode", density_mode, "--lam-eta-grid", lam_eta_grid,
                        "--out", str(out)],
                       capture_output=True, text=True)
    log = r.stdout + r.stderr
    if r.returncode == 0:
        # rmse.py refits a plain linear readout per candidate's radials: for a density
        # run only "init" and "radials_only" are meaningful there (a density
        # candidate's radials without its eta/d would be a misleading score), so
        # restrict to those two; rmse_npz.py evaluates the model.npz itself (which
        # for a density candidate carries the frozen FSModel term), so it runs on
        # the top-level model.npz AND every candidate's */model.npz.
        if density != "none":
            cands = ["--cand", "init", "--cand", str(out / "radials_only" / "rnl_Wnlq.npy")]
        else:
            cands = ["--cand", "init"] + sum((["--cand", str(p)] for p in sorted(out.glob("lam_*/rnl_Wnlq.npy"))), [])
        r2 = subprocess.run(["python", "/ace-jax/bench/learn_radial/rmse.py", *args, *cands,
                             "--out", str(out / "rmse.json")], capture_output=True, text=True)
        log += "\n== rmse ==\n" + r2.stdout + r2.stderr
        npz_paths = [out / "model.npz"] + sorted(out.glob("*/model.npz"))
        r3 = subprocess.run(["python", "/ace-jax/bench/learn_radial/rmse_npz.py",
                             *_drop(args, "--r0", "--n-q"),
                             *sum((["--model-npz", str(p)] for p in npz_paths), []),
                             "--out", str(out / "rmse_npz.json")],
                            capture_output=True, text=True)
        log += "\n== rmse_npz ==\n" + r3.stdout + r3.stderr
    files = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*")
             if p.is_file() and p.suffix in (".json", ".npy")} if out.exists() else {}
    return {"log": log, "files": files, "returncode": r.returncode}


@app.local_entrypoint()
def main(system: str = "sige", steps: str = "100,200,400", lam_grid: str = "0.1",
         reprofile_every: int = 50, n_q: int = 12, mults: str = "1", out: str = "runs/modal",
         density: str = "none", n_density: int = 1, density_mode: str = "joint", lam_eta_grid: str = "0"):
    # n_density, not P: Modal lower-cases CLI flags, so a `P` parameter is unreachable as --P
    P = n_density
    budgets = [int(s) for s in steps.split(",")]
    calls = [(system, s, lam_grid, reprofile_every, n_q, float(m), density, P, density_mode, lam_eta_grid)
            for s in budgets for m in mults.split(",")]
    for (sy, s, _, _, _, m, dn, _, _, _), res in zip(calls, learn.starmap(calls)):
        suffix = f"_{dn}{P}" if dn != "none" else ""
        d = pathlib.Path(out) / ((f"{sy}_s{s}" if m == 1.0 else f"{sy}_s{s}_m{m:g}") + suffix)
        d.mkdir(parents=True, exist_ok=True)
        (d / "log.txt").write_text(res["log"])
        for name, blob in res["files"].items():
            (d / name).parent.mkdir(parents=True, exist_ok=True)
            (d / name).write_bytes(blob)
        tail = [l for l in res["log"].splitlines() if "meV" in l or "gate" in l or "Error" in l]
        print(f"== {sy} steps={s} mult={m:g} rc={res['returncode']}\n" + "\n".join(tail[-8:]), flush=True)
