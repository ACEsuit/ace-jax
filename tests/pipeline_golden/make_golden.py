# tests/pipeline_golden/make_golden.py
"""Record the outputs of the UNCHANGED drivers (run.py and `ace-jax fit`) on the
fixtures, so the pipeline refactor can be checked for exact equivalence.
    uv run --extra gp python tests/pipeline_golden/make_golden.py
Re-run only if a behaviour change is intended (and say so in the commit)."""
import os, pathlib, platform, shutil, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIX = ROOT / "fixtures"
OUT = FIX / "pipeline_golden"
X = str(FIX / "si_tiny_train.xyz")
KEYS = ["--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key", "dft_virial"]
RUN = [sys.executable, str(ROOT / "bench/acegp_cantor/run.py"), "--model", str(FIX / "si_fitted.npz"),
       "--data", X, *KEYS, "--r0", "2.35", "--ntrain", "16", "--test-start", "16", "--ntest", "6",
       "--batch", "4", "--no-predict-train"]

SCENARIOS = {
    "run_linear_laplace": ("run", ["--arm", "linear", "--rungs", "map,laplace", "--map-steps", "15",
                                   "--n-draws", "4"]),
    "run_gp_pair_lbfgs": ("run", ["--arm", "gp", "--m-per-species", "6", "--density", "pair",
                                  "--rungs", "map", "--map-steps", "15"]),
    "run_gp_pca_hostcache_restarts": ("run", ["--arm", "gp", "--m-per-species", "6", "--density", "pca",
                                              "--pca-d", "8", "--lml", "host-cache", "--rungs", "map",
                                              "--map-steps", "10", "--map-restarts", "2"]),
    "run_linear_pops_auto": ("run", ["--arm", "linear", "--uq", "pops", "--rungs", "map",
                                     "--map-steps", "10", "--pops-ridge-grid", "1e-3,1e-6",
                                     "--pops-env-nf", "20"]),
    # OOD = Si_tiny configs 20-29 (all carry E/F/V): the unchanged run.py cannot score
    # configs without a virial (config 0), so the full file cannot be a golden
    "run_gp_adam_ood": ("run", ["--arm", "gp", "--m-per-species", "6", "--opt", "adam", "--rungs", "map",
                                "--map-steps", "10", "--ood", "{OOD}"]),
    "cli_map_laplace": ("cli", None),     # the trimmed tests/test_gp_cli.py settings
}


def run_argv(argv, out):
    """run.py command for a scenario; writes the scenario's OOD file into `out`."""
    if "{OOD}" in argv:
        from ase.io import read, write
        ood = out / "_ood.xyz"
        write(ood, read(X, "20:30"))
        argv = [str(ood) if x == "{OOD}" else x for x in argv]
    return [*RUN, *argv, "--out", str(out)]


def cli_argv(out):
    from ase.io import read, write
    cfgs = read(X, ":")
    tr, te = out / "_train.xyz", out / "_test.xyz"
    write(tr, cfgs[:12]); write(te, cfgs[12:16])
    return [sys.executable, "-m", "ace_jax.cli", "fit", "--model", str(FIX / "si_fitted.npz"),
            "--train", str(tr), "--test", str(te), *KEYS, "--configs-per-batch", "4",
            "--m-per-species", "6", "--rungs", "map,laplace", "--n-draws", "5",
            "--map-steps", "150", "--r0", "2.35", "--out", str(out)]


def platform_tag():
    """Where goldens were recorded: bit-level parity only holds on the same
    platform (BLAS/LAPACK round-off feeds the optimiser trajectories)."""
    return f"{platform.system()}-{platform.machine()}"


def main():
    env = dict(os.environ, JAX_ENABLE_X64="1", PYTHONPATH=str(ROOT))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PLATFORM").write_text(platform_tag() + "\n")
    for name, (driver, argv) in SCENARIOS.items():
        out = OUT / name
        shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
        cmd = cli_argv(out) if driver == "cli" else run_argv(argv, out)
        subprocess.run(cmd, check=True, env=env, cwd=ROOT)
        for junk in ("timings.json", "_train.xyz", "_test.xyz", "_ood.xyz", "gpu_mem_mib.log"):
            (out / junk).unlink(missing_ok=True)
        print("golden:", name, sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
