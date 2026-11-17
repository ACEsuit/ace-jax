"""Foundation MACE models in the Si accuracy-cost benchmark (family `mace_fm`): zero-shot, and fine-tuned on the
benchmark's training split.  Run under a mace-torch venv.

    python -m accuracy_cost.fit_mace_fm register SIZE     # zero-shot: copy the model, fit the energy shift
    python -m accuracy_cost.fit_mace_fm fit SIZE          # fine-tune (sweep entries with `finetune`)
    python -m accuracy_cost.fit_mace_fm export SIZE       # model-fdn.json for pair_style symmetrix/mace/foundation

Zero-shot rows are not fitted to the benchmark's data, with one exception.  A foundation model's total energies are
on its own reference (functional, pseudopotential, element reference energies), so they differ from the DFT labels
by about a constant per atom.  `register` records that constant: the mean of (E_DFT - E_model) / N over the
*training* split (`e_shift`, eV/atom; the isolated atom is left out).  The calculator adds it.  Forces, virials and
the QoIs (energy differences) do not depend on it.

Fine-tuned rows (`finetune` in the sweep entry) are mace_run_train's single-head ("naive") fine-tuning of every
weight from the entry's head, on the benchmark's training split with fit_mace.py's settings except E0: virials, the GAP-18 per-type weights, float64, 5% of train held out, --swa.  Only the learning rate and the
number of epochs are the entry's.  The architecture, and so the cutoff (MACE-MH-1: 6 A, two layers), is the
foundation model's: these rows do not follow the benchmark's 5 A rule.  E0 is the head's own Si reference energy
plus the zero-shot e_shift (`head_e0`), not the isolated-atom DFT energy: the fine-tuning then starts from the
zero-shot model's energies, not several eV per atom away from them.

The model file comes from $ACEJAX_AC_MACE_FM (a directory) or mace's cache, by the sweep entry's `model` name."""
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

from . import common
from .common import DATA

FAMILY = "mace_fm"
CACHE = pathlib.Path(os.environ.get("ACEJAX_AC_MACE_FM") or pathlib.Path.home() / ".cache" / "mace")


def source(name):
    for p in (CACHE / f"{name}.model", CACHE / (name.replace("-", "") + "model")):   # mace's cache drops punctuation
        if p.exists():
            return p
    raise FileNotFoundError(f"{name}: no model file in {CACHE} (set ACEJAX_AC_MACE_FM)")


def raw_calculator(path, head, device="cuda"):
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=str(path), device=device, default_dtype="float64", **({"head": head} if head else {}))


def calculator(path, device=None):
    """The scoring calculator of the run directory that holds `path`: the registered head, plus e_shift per atom."""
    import torch
    st = common.read_status(pathlib.Path(path).parent)
    calc = raw_calculator(path, st.get("head"), device or ("cuda" if torch.cuda.is_available() else "cpu"))
    shift, base = float(st.get("e_shift") or 0.0), type(calc)

    class Shifted(base):
        def calculate(self, atoms=None, *a, **k):
            super().calculate(atoms, *a, **k)
            n = len(atoms if atoms is not None else self.atoms)
            for key in ("energy", "free_energy"):
                if key in self.results:
                    self.results[key] = self.results[key] + shift * n
    calc.__class__ = Shifted
    return calc


def energy_shift(path, head, device="cuda"):
    """Mean (E_DFT - E_model) / N over the training split (the isolated atom left out)."""
    import torch
    from .split import read_atoms
    calc, d = raw_calculator(path, head, device), []
    for at in read_atoms(DATA / "train.xyz"):
        if at.info.get("config_type") != "isolated_atom":
            a = at.copy(); a.calc = calc
            d.append((at.info["energy"] - a.get_potential_energy()) / len(at))
    del calc; torch.cuda.empty_cache()             # fit() trains in a child process on the same GPU
    return float(np.mean(d)), len(d)


def register(size, device="cuda"):
    """Zero-shot: RUNS/mace_fm/<size>/model.model (a copy), status.json with head, e_shift and the parameter count."""
    import torch
    from .split import ensure_split
    e, d = common.entry(FAMILY, size), common.run_dir(FAMILY, size)
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "model.model").exists():
        shutil.copy(source(e["model"]), d / "model.model")
    m = torch.load(d / "model.model", weights_only=False, map_location="cpu")
    shift, n = energy_shift(d / "model.model", e.get("head"), device)
    common.write_status(d, family=FAMILY, size=size, status="ok", model=str(d / "model.model"), head=e.get("head"),
                        zero_shot=True, e_shift=shift, e_shift_frames=n, split_hash=ensure_split(),
                        n_params=int(sum(p.numel() for p in m.parameters())), fit_wall_s=0.0)
    return common.read_status(d)


def head_e0(path, head):
    """The Si reference energy of a head of the saved model (eV)."""
    import torch
    m = torch.load(path, weights_only=False, map_location="cpu")
    E = m.atomic_energies_fn.atomic_energies.detach().double().numpy()
    z = [int(x) for x in m.atomic_numbers].index(14)
    return float(E[list(m.heads).index(head), z] if E.ndim == 2 else E[z])


def command(e, d):
    w = json.loads((DATA / "weights_mace.json").read_text()) if (DATA / "weights_mace.json").exists() else {"Default": 1.0}
    ft = e["finetune"]
    return ["mace_run_train", "--name=model", f"--foundation_model={d / 'base.model'}",
            *([f"--foundation_head={e['head']}"] if e.get("head") else []), "--multiheads_finetuning=False",
            f"--train_file={e.get('train_file') or DATA / 'train.xyz'}", "--valid_fraction=0.05",
            "--energy_key=energy", "--forces_key=forces", "--virials_key=virial", "--loss=virials",
            f"--E0s={{14: {e['E0']}}}", f"--config_type_weights={json.dumps(w)}", "--batch_size=5",
            f"--lr={ft['lr']}", f"--max_num_epochs={ft['epochs']}", "--swa", "--ema", "--default_dtype=float64",
            f"--device={os.environ.get('ACEJAX_AC_DEVICE', 'cuda')}", f"--seed={e.get('seed', 0)}",
            f"--results_dir={d}", f"--model_dir={d}", f"--checkpoints_dir={d / 'ckpt'}", "--restart_latest",
            *ft.get("args", [])]


def fit(size, overrides=None):
    """Fine-tune: RUNS/mace_fm/<size>/model.model from base.model (a copy of the foundation model)."""
    import torch
    from .fit_mace import _path, subset
    from .split import ensure_split
    from .weights import write_all
    _path()
    h = ensure_split(); write_all(DATA)
    e, d = {**common.entry(FAMILY, size), **(overrides or {})}, common.run_dir(FAMILY, size)
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "base.model").exists():
        shutil.copy(source(e["model"]), d / "base.model")
    if e.get("n_train"):
        e["train_file"] = subset(d, e["n_train"])
    if not ((d / "model.model").exists() and (common.read_status(d) or {}).get("status") == "ok"):
        if not (d / "e0.json").exists():           # once: restarts of a preempted fit keep the same E0
            shift, n = energy_shift(d / "base.model", e.get("head"))
            (d / "e0.json").write_text(json.dumps({"head_e0": head_e0(d / "base.model", e["head"]), "e_shift": shift, "frames": n}))
        z = json.loads((d / "e0.json").read_text())
        e["E0"] = z["head_e0"] + z["e_shift"]
    st = common.fit_once(FAMILY, size, "model.model", command(e, d) if "E0" in e else [], split_hash=h)
    if st["status"] == "ok" and not st.get("n_params"):
        two = d / "model_stagetwo.model"          # --swa: the stage-two (energy-weighted) model is the one deployed
        if two.exists() and not (d / "model_stage1.model").exists():
            shutil.move(d / "model.model", d / "model_stage1.model"); shutil.copy(two, d / "model.model")
        m = torch.load(d / "model.model", weights_only=False, map_location="cpu")
        heads = [x for x in (getattr(m, "heads", None) or []) if x]
        common.write_status(d, **{**st, "n_params": int(sum(p.numel() for p in m.parameters())),
                                  "head": heads[0] if len(heads) == 1 and heads[0] != "Default" else None,
                                  "foundation": e["model"], "foundation_head": e.get("head"), "finetune": e["finetune"],
                                  "E0": e["E0"]})
        st = common.read_status(d)
    return st


def export_fdn(size, extract="symmetrix_extract_mace"):
    """model-fdn.json: the registered head, Si only, by Symmetrix v0.3.0's symmetrix_extract_mace (`extract`)."""
    d = common.run_dir(FAMILY, size)
    st = common.read_status(d)
    if not (d / "model-fdn.json").exists() or (d / "model-fdn.json").stat().st_mtime < (d / "model.model").stat().st_mtime:
        from .fit_mace import cpu_copy                # a fine-tuned model keeps CUDA tensors; the extractor's torch is CPU-only
        cmd = [extract, "--model", str(cpu_copy(d)), "--atomic-numbers", "14", "--output", str(d / "model-fdn.json"),
               *(["--head", st["head"]] if st.get("head") else [])]
        subprocess.run(cmd, check=True)
    return d / "model-fdn.json"


if __name__ == "__main__":
    what, sizes = sys.argv[1], sys.argv[2:]
    for s in sizes:
        print({"register": register, "fit": fit, "export": export_fdn}[what](s), flush=True)
