"""Big-cell OOD set v3 on GPU (gen/big3.py): the v2 cracks (K/K_G 1.1/1.2/1.3) and dissociated
edge/screw dislocations, made thick along the line (period > 2 rcut = 12.5 A) so the chemistry is
random along it; ~4k atoms each (R 29-34 A).  Two species realisations each; MACE Cantor properties
reused from v2 (defects/big2_props.json).  Resumes from defects/big3_mh1.xyz.

  MACE_MODEL=/path/mace-mh-1.model modal run --detach modal_big3.py
"""
import os
import pathlib

import modal

HOME = pathlib.Path.home()
ACEGP = pathlib.Path(os.environ.get("ACEGP_DATA", HOME / "acegp-data"))   # data + generated sets
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("mace-torch==0.3.16", "ase==3.23.0", "matscipy==1.1.1")
    .add_local_file(os.environ.get("MACE_MODEL", str(pathlib.Path.home() / "gits/SimpleGPpotential/models/mace-mh-1.model")), "/data/mace-mh-1.model")
    .add_local_file(str(pathlib.Path(__file__).parents[1] / "gen" / "big2.py"), "/root/big2.py")
    .add_local_file(str(pathlib.Path(__file__).parents[1] / "gen" / "big3.py"), "/root/big3.py")
)
app = modal.App("acegp-big3")
vol = modal.Volume.from_name("acegp-prod-out")


@app.function(gpu="B200", image=image, volumes={"/out": vol}, timeout=12 * 3600, memory=64 * 1024)   # A100-80GB OOMs on the 7.2k-atom edge cell (float64)
def run() -> str:
    import json, os, sys, warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "/root")
    import numpy as np
    from ase.io import write
    from mace.calculators import MACECalculator
    import big2
    import big3
    os.makedirs("/out/defects", exist_ok=True)
    logf = open("/out/defects/big3.log", "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush(); vol.commit()
    calc = MACECalculator(model_paths="/data/mace-mh-1.model", device="cuda", default_dtype="float64",
                          head="matpes_r2scan")
    rng = np.random.default_rng(27)
    if os.path.exists("/out/defects/big2_props.json"):          # resume: properties already measured
        p = json.load(open("/out/defects/big2_props.json")); a0, C, gamma = p["a0"], tuple(p["C"]), p["gamma111"]
        log(f"resume: a0 {a0:.4f}, C {C}, gamma111 {gamma:.4f}")
    else:
        log("MACE Cantor properties:")
        a0, C, gamma = big2.cantor_properties(calc, rng, log=log)
        log(f"mean: a0 {a0:.4f} A, C11/C12/C44 {C[0]:.1f}/{C[1]:.1f}/{C[2]:.1f} GPa, gamma111 {gamma:.4f} eV/A^2")
        json.dump({"a0": a0, "C": C, "gamma111": gamma}, open("/out/defects/big2_props.json", "w"), indent=1)
    from ase.io import read
    out = read("/out/defects/big3_mh1.xyz", ":") if os.path.exists("/out/defects/big3_mh1.xyz") else []
    todo = [(r, "crack", k) for r in range(2) for k in (1.1, 1.2, 1.3)] + \
           [(r, kind, None) for r in range(2) for kind in ("edge", "screw")]
    todo.sort(key=lambda t: (t[0], t[1] != "crack"))
    done = len(out) // 3
    log(f"{done} cells already done; {len(todo) - done} to go")
    for r, kind, k in todo[done:]:
        rng = np.random.default_rng(27 + 100 * r + (0 if k is None else int(10 * k)) + len(kind))
        if kind == "crack":
            out += big2.relax_label(big3.crack(rng, a0, C, gamma, k), calc, rng, log=log)
            write("/out/defects/big3_mh1.xyz", out, format="extxyz"); vol.commit()
            continue
        if True:
            at = big3.dislocation(rng, a0, C, kind)
            log(f"  {kind}: {len(at)} atoms, partial_distance {at.info['partial_distance']} x "
                f"glide {at.info['glide_distance']:.2f} A")
            out += big2.relax_label(at, calc, rng, log=log)
            write("/out/defects/big3_mh1.xyz", out, format="extxyz"); vol.commit()
    log(f"wrote {len(out)} configs")
    return f"wrote {len(out)} configs"


@app.local_entrypoint()
def main():
    print(run.remote())


@app.function(gpu="B200", image=image.add_local_file(str(pathlib.Path(__file__).parents[1] / "gen" / "big3_local.py"),
                                                     "/root/big3_local.py"),
              volumes={"/out": vol}, timeout=6 * 3600, memory=64 * 1024)
def extra(first: int = 2, last: int = 9) -> str:
    """Extra v3 CRACK realisations first..last (gen/big3_local.py's recipe and seeding) ->
    /out/defects/big3_cracks_r<first>-<last>.xyz.  A local 20 GB GPU runs out of memory on one
    3.3k-atom cell in float64 (needs ~27 GB), hence Modal."""
    import os
    import subprocess
    import time
    out = f"/out/defects/big3_cracks_r{first}-{last}.xyz"
    env = dict(os.environ, MACE_MODEL="/data/mace-mh-1.model", BIG2_PROPS="/out/defects/big2_props.json")
    with open(f"/out/defects/big3_cracks_r{first}-{last}.log", "a") as so:
        p = subprocess.Popen(["python", "-u", "/root/big3_local.py", out, "--first", str(first), "--last", str(last)],
                             stdout=so, stderr=subprocess.STDOUT, env=env, cwd="/root")
        while p.poll() is None:
            time.sleep(60); vol.commit()
    vol.commit()
    return f"realisations {first}-{last}: rc {p.returncode}"


@app.local_entrypoint()
def launch_extra(first: int = 2, last: int = 9, per: int = 2):
    calls = [extra.spawn(r, min(r + per - 1, last)) for r in range(first, last + 1, per)]
    for c in calls:
        print(c.get())
