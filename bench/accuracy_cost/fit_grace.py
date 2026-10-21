"""GRACE trained from scratch by gracemaker (tensorpotential 0.6.2.1): the from-scratch defaults of
`gracemaker -t` stand (Adam, lr 0.008 with EMA 0.99 and clipnorm 1, cosine decay to lr/12, huber loss
with delta 0.01, energy 16 / forces 32 / stress 128, batch 16, 50k updates, 5% of train held out to
pick the best checkpoint), apart from the shared cutoff, E0 (reference_energy), the GAP-18 per-type
weights and float64 parameters.  Families: `grace` (GRACE-1L ladder), `grace_fs` (GRACE-FS) and
`grace_abl` (one-change ablations of a `grace` size: `base`, plus `readout`, `freeze_radial` or kwargs).
Run under the grace venv (envs/grace; data.md), from bench/:

    python -m accuracy_cost.fit_grace grace c32 --seed 0 [--overrides JSON]

Each seed is its own run directory, runs/<family>/<size>_s<seed> (gracemaker writes seed/<seed>/ in it),
scored and registered as size "<size>_s<seed>".

Weights (weights.py's GAP-18 sigmas, as pacemaker's 1/sigma^2 but relative to the default sigma, so
gracemaker's own loss weights set the E:F:S balance): energy_weight (sE_d/sE_t)^2, force_weight
(sF_d/sF_t)^2 / (3N) per atom (gracemaker's default per-structure force normalisation times the type
ratio), virial_weight (sV_d/sV_t)^2 on frames with a DFT virial and 0 on the others.  gracemaker divides
each loss term by the batch's summed weights, so only ratios matter.  Stress: -virial / volume (ASE
Voigt order), fitted by gracemaker's stress loss (eV/A^3)."""
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

from . import common
from .common import DATA, E0_SI, RCUT

FAMILIES = ("grace", "grace_fs", "grace_abl")
DEFAULT_FIT = {        # tensorpotential/cli/wizard.py's from-scratch answers (Adam), written out so the run records them
    "loss": {"energy": {"weight": 16, "type": "huber", "delta": 0.01},
             "forces": {"weight": 32, "type": "huber", "delta": 0.01},
             # normalize_by_samples False: gracemaker divides the stress term by the batch's summed virial weights,
             # 0/0 = NaN in a batch with no virials (surfaces, amorph, sp...); dataframe() divides the weights by the
             # expected batch sum instead, so a typical batch sees the same normalisation
             "stress": {"weight": 128.0, "type": "huber", "delta": 0.01, "normalize_by_samples": False}},
    "target_total_updates": 50000,
    "optimizer": "Adam",
    "opt_params": {"learning_rate": 0.008, "use_ema": True, "ema_momentum": 0.99, "weight_decay": None, "clipnorm": 1.0},
    "scheduler": "cosine_decay", "scheduler_params": {"minimum_learning_rate": 0.008 / 12},
    "batch_size": 16, "test_batch_size": 64,
    "jit_compile": True, "train_max_n_buckets": "auto", "test_max_n_buckets": "auto", "auto_bucket_max_padding": 0.3,
    "checkpoint_freq": 10, "progressbar": False, "train_shuffle": True,
}
MODEL_KEYS = ("lmax", "n_rad_max", "prod_func_n_max", "max_order", "n_mlp_dens", "n_rad_base", "embedding_size",
              "basis_type", "Lmax", "max_sum_l", "fs_parameters")


def resolve(family, size, sweep=None):
    """The sweep entry; a `grace_abl` entry takes its `base` grace entry's kwargs, then its own."""
    sweep = sweep or common.load_sweep()
    e = dict(common.entry(family, size, sweep))
    if family == "grace_abl":
        b = dict(common.entry("grace", e["base"], sweep))
        e = {**{k: v for k, v in b.items() if k not in ("size", "n_params")}, **e}
    return e


def dataframe(atoms, d, per, batch_size=DEFAULT_FIT["batch_size"]):
    """gracemaker's pandas input: ase_atoms, energy, forces, stress and the three weight columns.  The virial
    weights are divided by 6 x their expected sum over a batch (see DEFAULT_FIT's stress loss)."""
    import pandas as pd
    rows = []
    for a in atoms:
        sE, sF, sV = per.get(a.info["config_type"], d)
        b = a.copy(); b.calc = None
        n, V = len(b), b.get_volume()
        vir = a.info.get("virial")
        if vir is not None:
            s = -np.reshape(vir, (3, 3)) / V
            stress = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        # no virial: zero stress at weight 0 (gracemaker 0.6.2.1 multiplies np.array(None) before its None check)
        rows.append({"ase_atoms": b, "energy": a.info["energy"], "forces": np.asarray(a.arrays["forces"]),
                     "stress": stress if vir is not None else np.zeros(6), "config_type": a.info["config_type"],
                     "energy_weight": (d[0] / sE) ** 2,
                     "force_weight": np.full(n, (d[1] / sF) ** 2 / (3 * n)),
                     "virial_weight": (d[2] / sV) ** 2 if vir is not None else 0.0})
    df = pd.DataFrame(rows)
    df["virial_weight"] /= 6 * batch_size * df["virial_weight"].mean()
    return df


def input_yaml(e, seed):
    kw = {k: e[k] for k in MODEL_KEYS if k in e}
    if e.get("preset") == "FS":
        pot = {"preset": "FS", "kwargs": kw}
    else:                                            # every GRACE-1L size goes through grace_custom (readout switch)
        pot = {"custom": "grace_custom.grace_1l", "kwargs": {**kw, "readout": e.get("readout", "mlp")}}
    fit = json.loads(json.dumps(DEFAULT_FIT))
    for k in ("target_total_updates", "batch_size"):
        if k in e:
            fit[k] = e[k]
    if e.get("freeze_radial"):
        fit["trainable_variable_names"] = e["trainable"]    # sweep lists the non-radial prefixes (see sweep.yaml)
    return {"seed": seed, "cutoff": RCUT,
            "data": {"filename": "train.pkl.gz", "test_size": 0.05, "reference_energy": {"Si": E0_SI},
                     "save_dataset": False},
            "potential": {**pot, "param_dtype": "float64", "scale": True},
            "fit": fit}


def _tool(name):
    return str(pathlib.Path(sys.executable).parent / name)


def model_dir(d, seed):
    return pathlib.Path(d) / "seed" / str(seed)


def n_params(path_yaml):
    """(total, trainable-by-default) parameter counts of a gracemaker model.yaml, built in float64."""
    import tensorflow as tf
    from tensorpotential.instructions.base import load_instructions
    from tensorpotential.tpmodel import TPModel
    tp = TPModel(load_instructions(str(path_yaml))); tp.build(tf.float64)
    return int(sum(int(np.prod(v.shape)) for v in tp.trainable_variables))


def export(d, seed, e):
    """FS: the C++ YAML (pair_style grace/fs); GRACE-1L: the Kokkos .npz (pair_style grace/1l/kk) where the
    model fits its caps.  final_model/ (the TF SavedModel, pair_style grace) is gracemaker's own output."""
    m = model_dir(d, seed); ck = str(m / "checkpoints" / "checkpoint.best_test_loss")
    base = [_tool("grace_utils"), "-p", str(m / "model.yaml"), "-c", ck]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}     # CPU: the GPU may be held by a running fit
    out = {}
    if e.get("preset") == "FS":
        r = subprocess.run(base + ["export", "-sf", "-n", str(m / "FS_model.yaml")], capture_output=True, text=True, env=env)
        out["fs_yaml"] = str(m / "FS_model.yaml") if r.returncode == 0 else None
        out["fs_export_error"] = None if r.returncode == 0 else r.stderr[-2000:]
    else:
        r = subprocess.run(base + ["export_kokkos", "-o", str(m / "model_kk.npz")], capture_output=True, text=True, env=env)
        out["kokkos_npz"] = str(m / "model_kk.npz") if r.returncode == 0 else None
        out["kokkos_export_error"] = None if r.returncode == 0 else (r.stderr or r.stdout)[-2000:]
    return out


def calculator(path):
    """TPCalculator on the SavedModel (float64) with E0_SI per atom added back: gracemaker fits
    energy - N E0 (reference_energy), so its model predicts that; the other families predict E."""
    from tensorpotential.calculator import TPCalculator

    class _GRACE(TPCalculator):
        def calculate(self, atoms=None, properties=None, system_changes=None):
            super().calculate(atoms, properties, system_changes)
            for k in ("energy", "free_energy"):
                if k in self.results:
                    self.results[k] = self.results[k] + len(self.atoms) * E0_SI
            if "energies" in self.results:
                self.results["energies"] = self.results["energies"] + E0_SI
    return _GRACE(str(path))


def fit(family, size, seed=0, overrides=None):
    from .split import XML, ensure_split, read_atoms
    from .weights import command_line, gap_sigmas
    import yaml
    h = ensure_split()
    e = {**resolve(family, size), **(overrides or {})}
    name = f"{size}_s{seed}"
    d = common.run_dir(family, name); d.mkdir(parents=True, exist_ok=True)
    ats = [a for a in read_atoms(DATA / "train.xyz") if a.info["config_type"] != "isolated_atom"]  # no neighbours: NaN loss
    if e.get("n_train"):
        ats = ats[::max(1, len(ats) // e["n_train"])][:e["n_train"]]        # strided: the file is sorted by config_type
    if not (d / "train.pkl.gz").exists():
        dataframe(ats, *gap_sigmas(command_line(DATA / XML)), e.get("batch_size", DEFAULT_FIT["batch_size"])).to_pickle(d / "train.pkl.gz", compression="gzip")
    shutil.copy(common.HERE / "grace_custom.py", d / "grace_custom.py")
    (d / "input.yaml").write_text(yaml.safe_dump(input_yaml(e, seed), sort_keys=False))
    rel = f"seed/{seed}/final_model"
    st = common.fit_once(family, name, rel, [_tool("gracemaker"), "input.yaml"], split_hash=h)
    if st["status"] == "ok" and not (st.get("n_params") and (st.get("kokkos_npz") or st.get("fs_yaml"))):
        st = finish(d, seed, e, st)
    return st


def finish(d, seed, e, st):
    """n_params and the LAMMPS export, recorded in status.json (the fit's host kept); re-run when an export failed."""
    m = model_dir(d, seed)
    common.write_status(d, **{**{k: v for k, v in st.items() if k != "host"}, "n_params": n_params(m / "model.yaml"),
                              "seed": seed, "sweep_entry": e, **export(d, seed, e),
                              **({"host": st["host"]} if st.get("host") else {})})
    return common.read_status(d)


PARITY_TOL = (1e-10, 1e-9)     # |dE|/atom (eV), max|dF| (eV/A): both sides float64 (scaling.parity's acejax bound)


def lammps_model(st):
    """(run_lammps style, model file) of a finished fit: GRACE-FS's C++ YAML, else the 1L Kokkos .npz."""
    if st.get("fs_yaml"):
        return "grace-fs", st["fs_yaml"]
    if st.get("kokkos_npz"):
        return "grace", st["kokkos_npz"]
    raise ValueError(f"{st['family']}/{st['size']}: no LAMMPS export ({st.get('kokkos_export_error') or st.get('fs_export_error')})")


def lammps_parity(family, name, lmp, device="cpu", cells=3, n_atoms=64, rattle=0.05, work=None):
    """The timing gate's check (timing.gate_cells: rattled 64-atom diamond cells) for one fit: LAMMPS `run 0`
    against TPCalculator on the SavedModel, both without E0 (LAMMPS evaluates the model as trained)."""
    sys.path.insert(0, str(common.HERE.parent))
    from scaling.run_lammps import finished, kokkos_args, lammps_input, read_dump_forces, read_pe
    from scaling.structures import supercell
    from ase.io import write
    from tensorpotential.calculator import TPCalculator
    d = common.run_dir(family, name); st = common.read_status(d)
    style, model = lammps_model(st)
    calc = TPCalculator(st["model"])
    work = pathlib.Path(work or d / f"parity-{device}"); rows = []
    for k in range(cells):
        at = supercell("Si", n_atoms, seed=k, rattle=rattle)
        a = at.copy(); a.calc = calc
        E0, F0 = a.get_potential_energy(), a.get_forces()
        w = work / f"cell{k}"; w.mkdir(parents=True, exist_ok=True)
        write(w / "x.data", at, format="lammps-data", specorder=["Si"], masses=True)
        (w / "in.parity").write_text(lammps_input(style, model, ["Si"], w / "x.data", device, 0,
                                                  dump=str(w / "f.dump"), warmup=0))
        p = subprocess.run([lmp, "-in", "in.parity", "-log", "log.lammps", "-nocite", *kokkos_args(style, device)],
                           cwd=w, capture_output=True, text=True, timeout=1800)
        log = (w / "log.lammps").read_text() if (w / "log.lammps").exists() else ""
        if not finished(p.returncode, log):
            raise RuntimeError(f"lammps exit {p.returncode}: {(p.stderr or p.stdout)[-600:]}")
        E1, F1 = read_pe(log), read_dump_forces(w / "f.dump")
        dE, dF = abs(E1 - E0) / len(at), float(np.abs(F1 - F0).max())
        rows.append({"cell": k, "E_ase": E0, "E_lammps": E1, "dE_per_atom": dE, "max_dF": dF, "max_F": float(np.abs(F0).max()),
                     "ok": dE <= PARITY_TOL[0] and dF <= PARITY_TOL[1]})
    out = {"family": family, "size": name, "style": style, "model": model, "device": device, "lmp": str(lmp),
           "tol": PARITY_TOL, "status": "parity_ok" if all(r["ok"] for r in rows) else "parity_fail", "cells": rows}
    (work / "parity.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=FAMILIES)
    ap.add_argument("size", nargs="+")
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--overrides", default="", help="JSON sweep-entry overrides (smoke: n_train, target_total_updates)")
    ap.add_argument("--parity", metavar="LMP", help="instead of fitting: the 64-atom LAMMPS parity check of finished fits")
    ap.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    a = ap.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    if a.parity:
        for s in a.size:
            for k in a.seed:
                r = lammps_parity(a.family, f"{s}_s{k}", a.parity, a.device)
                print(r["status"], *(f"cell{c['cell']}: dE/atom {c['dE_per_atom']:.2e} max|dF| {c['max_dF']:.2e}" for c in r["cells"]))
        sys.exit(0)
    for s in a.size:
        for k in a.seed:
            print(a.family, s, k, fit(a.family, s, k, json.loads(a.overrides) if a.overrides else None)["status"], flush=True)
