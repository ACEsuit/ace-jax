"""GRACE foundation models in the Si accuracy-cost benchmark (family `grace_fm`): zero-shot, and fine-tuned on the
benchmark's training split.  Run under the grace venv (envs/grace), from bench/:

    grace_models download NAME --kokkos; grace_models checkpoint NAME      # once, where there is network
    python -m accuracy_cost.fit_grace_fm register SIZE                     # zero-shot
    python -m accuracy_cost.fit_grace_fm fit SIZE                          # fine-tune (sweep entries with `finetune`)

The models come from gracemaker's cache ($GRACE_CACHE, default ~/.cache/grace), by the sweep entry's `model` name.
Each row keeps its own copy in its run directory: final_model/ (the TF SavedModel, the scoring model and the timing
gate's reference) and model_kk.npz (grace-runtime's pair_style grace/runtime).

As for mace_fm (fit_mace_fm.py), a model's energies are on its own reference.  `e_shift` (eV/atom) is the mean of
(E_DFT - E_model) / N over the training split, the isolated atom left out; the calculator adds it.  It is recorded
for fine-tuned rows too, where gracemaker's `shift: auto` has aligned the model already and it is close to zero.

Fine-tuning is gracemaker's own recipe (`gracemaker -t`, "finetune foundation model"): Adam at learning rate 0.001
with cosine decay, `reduce_elements` and `shift: auto`, every parameter updated ("naive").  The loss, the GAP-18
per-type weights, virials and the 5% hold-out are fit_grace.py's.  The architecture and its 6 A cutoff are the
foundation model's: these rows do not follow the benchmark's 5 A rule."""
import json
import os
import pathlib
import shutil
import sys

import numpy as np

from . import common
from .common import DATA
from .fit_grace import DEFAULT_FIT, _tool, dataframe, export, model_dir, n_params

FAMILY = "grace_fm"
CACHE = pathlib.Path(os.environ.get("GRACE_CACHE") or pathlib.Path.home() / ".cache" / "grace")
FT_LR = 0.001                                    # the wizard's fine-tuning learning rate (from scratch: 0.008)


def raw_calculator(path):
    from tensorpotential.calculator import TPCalculator
    return TPCalculator(str(path))


def calculator(path):
    """The scoring calculator of the run directory that holds `path`: TPCalculator plus e_shift per atom."""
    from tensorpotential.calculator import TPCalculator
    d = next(p for p in pathlib.Path(path).parents if (p / "status.json").exists())
    shift = float((common.read_status(d) or {}).get("e_shift") or 0.0)

    class _GRACE(TPCalculator):
        def calculate(self, atoms=None, properties=None, system_changes=None):
            super().calculate(atoms, properties, system_changes)
            for k in ("energy", "free_energy"):
                if k in self.results:
                    self.results[k] = self.results[k] + len(self.atoms) * shift
            if "energies" in self.results:
                self.results["energies"] = self.results["energies"] + shift
    return _GRACE(str(path))


def train_atoms():
    from .split import read_atoms
    return [a for a in read_atoms(DATA / "train.xyz") if a.info["config_type"] != "isolated_atom"]


def energy_shift(path):
    calc, d = raw_calculator(path), []
    for at in train_atoms():
        a = at.copy(); a.calc = calc
        d.append((at.info["energy"] - a.get_potential_energy()) / len(at))
    return float(np.mean(d)), len(d)


def register(size):
    """Zero-shot: copies of the cached SavedModel and kokkos.npz, status.json with e_shift and the parameter count."""
    from .split import ensure_split
    e, d = common.entry(FAMILY, size), common.run_dir(FAMILY, size)
    d.mkdir(parents=True, exist_ok=True)
    src = CACHE / e["model"]
    if not (d / "final_model").exists():
        shutil.copytree(src, d / "final_model", ignore=shutil.ignore_patterns("kokkos.npz"))
    if not (d / "model_kk.npz").exists():
        shutil.copy(src / "kokkos.npz", d / "model_kk.npz")
    shift, n = energy_shift(d / "final_model")
    yml = CACHE / "checkpoints" / e["model"] / "model.yaml"
    common.write_status(d, family=FAMILY, size=size, status="ok", model=str(d / "final_model"),
                        kokkos_npz=str(d / "model_kk.npz"), zero_shot=True, e_shift=shift, e_shift_frames=n,
                        split_hash=ensure_split(), n_params=n_params(yml) if yml.exists() else None, fit_wall_s=0.0,
                        foundation=e["model"])
    return common.read_status(d)


def input_yaml(e, seed=0):
    ft, fit = e["finetune"], json.loads(json.dumps(DEFAULT_FIT))
    lr = ft.get("lr", FT_LR)
    fit["opt_params"]["learning_rate"] = lr
    fit["scheduler_params"] = {"minimum_learning_rate": lr / 12}
    fit["target_total_updates"] = ft["updates"]
    fit["eval_init_stats"] = True
    if "batch_size" in ft:
        fit["batch_size"] = ft["batch_size"]
    return {"seed": seed, "cutoff": 6.0,          # not used: the foundation model's own
            "data": {"filename": "train.pkl.gz", "test_size": 0.05, "save_dataset": False,
                     "reference_energy": 0},          # none subtracted: `shift: auto` aligns the model instead
            "potential": {"finetune_foundation_model": e["model"], "reduce_elements": True, "shift": "auto"},
            "fit": fit}


def fit(size, overrides=None, seed=0):
    import yaml
    from .split import XML, ensure_split
    from .weights import command_line, gap_sigmas
    h = ensure_split()
    e, d = {**common.entry(FAMILY, size), **(overrides or {})}, common.run_dir(FAMILY, size)
    seed = e.get("seed", seed)                    # the sweep entry's (sizes "<size>_s<k>"), else the argument
    d.mkdir(parents=True, exist_ok=True)
    ats = train_atoms()
    if e.get("n_train"):
        ats = ats[::max(1, len(ats) // e["n_train"])][:e["n_train"]]
    y = input_yaml(e, seed)
    if not (d / "train.pkl.gz").exists():
        dataframe(ats, *gap_sigmas(command_line(DATA / XML)), y["fit"]["batch_size"]).to_pickle(d / "train.pkl.gz", compression="gzip")
    (d / "input.yaml").write_text(yaml.safe_dump(y, sort_keys=False))
    st = common.fit_once(FAMILY, size, f"seed/{seed}/final_model", [_tool("gracemaker"), "input.yaml"], split_hash=h)
    if st["status"] == "ok" and not (st.get("n_params") and st.get("kokkos_npz")):
        m = model_dir(d, seed)
        shift, n = energy_shift(m / "final_model")
        common.write_status(d, **{**{k: v for k, v in st.items() if k != "host"}, "n_params": n_params(m / "model.yaml"),
                                  "seed": seed, "sweep_entry": e, **export(d, seed, e), "e_shift": shift, "e_shift_frames": n,
                                  "foundation": e["model"], **({"host": st["host"]} if st.get("host") else {})})
        st = common.read_status(d)
    return st


if __name__ == "__main__":
    what, sizes = sys.argv[1], sys.argv[2:]
    for s in sizes:
        print({"register": register, "fit": fit}[what](s), flush=True)
