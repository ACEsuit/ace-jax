"""Leave-one-realisation-out `aj calibrate` of a bench365 ARD run (README Step 4 item 4), as its own app so
the deployed acegp-bench365 is untouched.  Per hold-out pair: calibrate on the other three v3 crack files,
then per-atom errors + served arrays on the held-out file (big3x_r<hold>_err.npz) and on big3_mh1.xyz
(big3_err.npz) in /out/<run>_cal_r<hold>/.

  modal deploy modal_calibrate.py && modal run modal_calibrate.py::launch [--run bench365_ard_ABblk]
"""
import pathlib

import modal
from modal_bench365 import image as _base, vol

HERE = pathlib.Path(__file__).parent
HOLDS = ("2-3", "4-5", "6-7", "8-9")
image = (_base.add_local_file(str(HERE / "big_err_core.py"), "/root/big_err_core.py")
         .add_local_file(str(HERE / "modal_bench365.py"), "/root/modal_bench365.py")   # so this module imports
         .add_local_file(str(HERE / "modal_calibrate.py"), "/root/modal_calibrate.py"))
app = modal.App("acegp-calibrate")


def u_files(hold: str) -> list:
    """The v3 crack files of the other realisation pairs, in pair order."""
    if hold not in HOLDS:
        raise ValueError(f"hold {hold!r} not in {HOLDS}")
    return [f"/out/defects/big3_cracks_r{p}.xyz" for p in HOLDS if p != hold]


@app.function(gpu="A100-40GB", image=image, volumes={"/out": vol}, timeout=4 * 3600)
def calibrate_loro(run: str, hold: str, mode: str = "per-group") -> str:
    import os
    import shutil
    import subprocess
    import sys
    import time
    sys.path.insert(0, "/root")
    import big_err_core
    if mode not in ("per-group", "append", "replace"):
        raise ValueError(mode)
    t0 = time.time()
    out = f"/out/{run}_cal_r{hold}"
    os.makedirs(out, exist_ok=True)
    u = f"/tmp/U_{hold}.xyz"
    nfr = big_err_core.concat_xyz(u_files(hold), u)
    shutil.copy(f"/out/{run}/model.npz", f"{out}/model.npz")
    # no --virial-key: calibrate scores forces only
    argv = ["python", "-m", "ace_jax.cli", "calibrate", "--model", f"/out/{run}/model.npz",
            "--posterior", f"/out/{run}/posterior.npz", "--data", u, "--energy-key", "mace_energy",
            "--force-key", "mace_force", "--out", f"{out}/posterior.npz"] + \
        ([] if mode == "per-group" else [f"--{mode}"])
    env = dict(os.environ, JAX_ENABLE_X64="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    p = subprocess.run(argv, capture_output=True, text=True, env=env)
    open(f"{out}/calibrate.log", "w").write(p.stdout + ("\n--- stderr ---\n" + p.stderr if p.stderr else ""))
    vol.commit()
    if p.returncode:
        raise RuntimeError(f"calibrate rc={p.returncode}: {p.stderr[-2000:]}")
    s1 = big_err_core.err_npz(f"{out}/model.npz", f"{out}/posterior.npz",
                              f"/out/defects/big3_cracks_r{hold}.xyz", f"{out}/big3x_r{hold}_err.npz")
    s2 = big_err_core.err_npz(f"{out}/model.npz", f"{out}/posterior.npz",
                              "/out/defects/big3_mh1.xyz", f"{out}/big3_err.npz")
    vol.commit()
    last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    return (f"{run} hold r{hold} [{mode}] U={nfr} frames; {last}; held-out: {s1}; mh1: {s2}; "
            f"{time.time() - t0:.0f} s")


@app.local_entrypoint()
def launch(run: str = "bench365_ard_ABblk", holds: str = ",".join(HOLDS), mode: str = "per-group"):
    f = modal.Function.from_name("acegp-calibrate", "calibrate_loro")
    for h in holds.split(","):
        print("spawned", h, f.spawn(run, h, mode).object_id)
