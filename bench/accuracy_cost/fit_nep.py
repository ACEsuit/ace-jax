"""NEP (neuroevolution potential, NEP4) trained with GPUMD's `nep` executable on the train split, on a GPU.

- Data: DATA/train.xyz and test.xyz rewritten in GPUMD's extxyz dialect (`lattice`, `energy`, `virial`,
  `weight`, `species:S:1:pos:R:3:force:R:3`).  Energies are relative to the isolated atom (E0_SI per
  atom): NEP trains in single precision and warns below -100 eV/atom (CASTEP Si is ~ -158).  `calculator`
  adds E0_SI back.
- Weights: GAP-18's per-config-type force sigmas as a per-structure `weight` = sigma_F,default / sigma_F, the
  square root of DATA/weights_mace.json (MACE multiplies the squared error by its weight, GPUMD by weight^2:
  main_nep/dataset.cu get_rmse_force).  Fits before this fix (n10-n120c and the n120b screens) wrote the MACE
  weight itself, i.e. (sigma_F,default / sigma_F)^4 on the squared error.  An entry with `weights: none`
  writes weight 1.  NEP's own lambda_e / lambda_f / lambda_v (1, 1, 0.1) unless the entry sets them.
- Cutoff: 5 A radial and angular (RCUT, every family) unless the entry sets `cutoff`; the sweep entry sets
  n_max, basis_size, l_max, neuron (and optionally generation, batch, cutoff, lambda_1, lambda_2).

Evaluation (test errors, QoIs) and LAMMPS (`pair_style nep`, NEP_CPU's USER-NEP) both run NEP_CPU in
float64; GPUMD's own GPU inference is mixed precision and is not timed here (docs/dev/accuracy-cost-results.md).
$ACEJAX_AC_NEP names the nep executable (default `nep` on PATH)."""
import json
import os
import re
import shutil

import numpy as np

from . import common
from .common import DATA, E0_SI, RCUT

NEP_DEFAULTS = {"version": 4, "generation": 100000, "batch": 1000}


def nep_in(e):
    """nep.in for a sweep entry: n_max "R A", basis_size "R A", l_max "3b [4b [5b]]", neuron N; optionally
    cutoff "R A" (default RCUT both), lambda_1 / lambda_2 (GPUMD's default scales with the parameter count),
    population (SNES; default 50), the loss weights lambda_e / lambda_f / lambda_v (default 1, 1, 0.1) and
    import_q_scaler (1: keep nep.txt's descriptor scaling instead of recomputing it from the training set at
    generation 0; main_nep/fitness.cu).  A batch at least the training-set size is full batch."""
    reg = [f"{k} {e[k]:g}" for k in ("lambda_1", "lambda_2", "population", "lambda_e", "lambda_f", "lambda_v",
                                     "import_q_scaler") if k in e]
    return "\n".join([f"version {e.get('version', NEP_DEFAULTS['version'])}", "type 1 Si",
                      f"cutoff {e.get('cutoff', f'{RCUT:g} {RCUT:g}')}",
                      f"n_max {e['n_max']}", f"basis_size {e['basis_size']}", f"l_max {e['l_max']}",
                      f"neuron {e['neuron']}", *reg, f"batch {e.get('batch', NEP_DEFAULTS['batch'])}",
                      f"generation {e.get('generation', NEP_DEFAULTS['generation'])}", "prediction 0"]) + "\n"


def _fmt(v):
    return " ".join(f"{float(x):.10f}" for x in np.ravel(v))


def write_gpumd_xyz(path, atoms, weights):
    """GPUMD's extxyz: energy relative to E0_SI per atom, total virial (eV; symmetric, so row- vs
    column-major does not matter), a per-structure weight from the config type."""
    with open(path, "w") as f:
        for at in atoms:
            n, info = len(at), at.info
            e = info["energy"] - n * E0_SI
            fields = [f'lattice="{_fmt(at.cell.array)}"', f"energy={e:.10f}"]
            if "virial" in info:
                fields.append(f'virial="{_fmt(info["virial"])}"')
            fields.append(f"weight={weights.get(info.get('config_type'), weights['Default']):.6g}")
            fields.append("properties=species:S:1:pos:R:3:force:R:3")
            F = at.arrays["forces"]
            f.write(f"{n}\n{' '.join(fields)}\n")
            for s, p, fv in zip(at.get_chemical_symbols(), at.positions, F):
                f.write(f"{s} {_fmt(p)} {_fmt(fv)}\n")


def prepare(d, e):
    from .split import read_atoms
    w = {k: float(np.sqrt(v)) for k, v in json.loads((DATA / "weights_mace.json").read_text()).items()}
    if e.get("weights") == "none":
        w = {"Default": 1.0}
    d.mkdir(parents=True, exist_ok=True)
    write_gpumd_xyz(d / "train.xyz", read_atoms(DATA / "train.xyz"), w)
    write_gpumd_xyz(d / "test.xyz", read_atoms(DATA / "test.xyz"), w)
    (d / "nep.in").write_text(nep_in(e))


def n_params(log):
    """GPUMD prints the trainable parameter count while it sets up the model."""
    m = re.findall(r"total number of parameters to be optimized\s*=\s*(\d+)", log)   # main_nep/parameters.cu
    return int(m[-1]) if m else None


def fit(size):
    """Train to the entry's `generation` (total) in jobs of at most `chunk` generations: a run short of its total
    resumes from nep.restart (GPUMD reads it when present; SNES's step sizes are fixed, so a resumed run continues
    the same search).  status.json counts `generations`; fits from before it was recorded ran NEP_DEFAULTS'.
    An entry with `init: <size>` is a later stage of a multi-stage fit (the GPUMD developers' two-stage recipe:
    force-weighted, then balanced): its first job starts from that fit's nep.restart and nep.txt, and its
    `generations` and `fit_wall_s` count this stage only."""
    from .split import ensure_split
    h = ensure_split()
    e, d = common.entry("nep", size), common.run_dir("nep", size)
    total = int(e.get("generation", NEP_DEFAULTS["generation"]))
    prev = common.read_status(d) or {}
    done = prev.get("generations", NEP_DEFAULTS["generation"]) if prev.get("status") == "ok" else 0
    if prev.get("status") == "ok" and done >= total:
        return prev
    if not done:
        prepare(d, e)
        if "init" in e:                                           # a later stage: resume the SNES search of `init`
            src = common.run_dir("nep", e["init"])
            if (common.read_status(src) or {}).get("status") != "ok":
                raise RuntimeError(f"nep/{size} starts from nep/{e['init']}, which has no finished fit")
            for f in ("nep.restart", "nep.txt"):
                shutil.copy2(src / f, d / f)
    run = min(int(e.get("chunk", total)), total - done)
    (d / "nep.in").write_text(nep_in({**e, "generation": run}))
    if done:
        (d / "status.json").unlink()                              # fit_once runs only without an ok status
    st = common.fit_once("nep", size, "nep.txt", [os.environ.get("ACEJAX_AC_NEP", "nep")], split_hash=h)
    if st["status"] == "ok":
        st.update(generations=done + run, fit_wall_s=st["fit_wall_s"] + prev.get("fit_wall_s", 0.0) * bool(done),
                  n_params=prev.get("n_params") or n_params((d / "fit.log").read_text()), e0_per_atom=E0_SI,
                  **({"init": e["init"]} if "init" in e else {}))
        common.write_status(d, **{k: v for k, v in st.items() if k != "host"})
    return st


def calculator(path):
    """NEP_CPU (calorine, float64) with E0_SI per atom added back: the energies the other families predict."""
    from calorine.calculators import CPUNEP

    class _NEP(CPUNEP):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.implemented_properties.append("free_energy")   # CPUNEP.__init__ sets the list per instance

        def calculate(self, atoms=None, properties=None, system_changes=None):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] = self.results["energy"] + len(self.atoms) * E0_SI
            self.results["free_energy"] = self.results["energy"]   # CPUNEP has none; ASE's cell filters ask for it
            if "energies" in self.results:
                self.results["energies"] = self.results["energies"] + E0_SI
    return _NEP(str(path))


if __name__ == "__main__":
    import sys
    for s in sys.argv[1:]:
        print(json.dumps(fit(s)))
