"""NEP (neuroevolution potential, NEP4) trained with GPUMD's `nep` executable on the train split, on a GPU.

- Data: DATA/train.xyz and test.xyz rewritten in GPUMD's extxyz dialect (`lattice`, `energy`, `virial`,
  `weight`, `species:S:1:pos:R:3:force:R:3`).  Energies are relative to the isolated atom (E0_SI per
  atom): NEP trains in single precision and warns below -100 eV/atom (CASTEP Si is ~ -158).  `calculator`
  adds E0_SI back.
- Weights: GAP-18's per-config-type force sigmas as a per-structure `weight`, as fit_mace does
  (DATA/weights_mace.json, weights.mace); NEP's own lambda_e / lambda_f / lambda_v otherwise (1, 1, 0.1).
- Cutoff: 5 A radial and angular (RCUT, every family); the sweep entry sets n_max, basis_size, l_max,
  neuron (and optionally generation, batch).

Evaluation (test errors, QoIs) and LAMMPS (`pair_style nep`, NEP_CPU's USER-NEP) both run NEP_CPU in
float64; GPUMD's own GPU inference is mixed precision and is not timed here (docs/dev/accuracy-cost-results.md).
$ACEJAX_AC_NEP names the nep executable (default `nep` on PATH)."""
import json
import os
import re

import numpy as np

from . import common
from .common import DATA, E0_SI, RCUT

NEP_DEFAULTS = {"version": 4, "generation": 100000, "batch": 1000}


def nep_in(e):
    """nep.in for a sweep entry: n_max "R A", basis_size "R A", l_max "3b [4b [5b]]", neuron N."""
    return "\n".join([f"version {e.get('version', NEP_DEFAULTS['version'])}", "type 1 Si", f"cutoff {RCUT:g} {RCUT:g}",
                      f"n_max {e['n_max']}", f"basis_size {e['basis_size']}", f"l_max {e['l_max']}",
                      f"neuron {e['neuron']}", f"batch {e.get('batch', NEP_DEFAULTS['batch'])}",
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
    w = json.loads((DATA / "weights_mace.json").read_text())     # weights.mace: per config type, as fit_mace
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
    the same search).  status.json counts `generations`; fits from before it was recorded ran NEP_DEFAULTS'."""
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
    run = min(int(e.get("chunk", total)), total - done)
    (d / "nep.in").write_text(nep_in({**e, "generation": run}))
    if done:
        (d / "status.json").unlink()                              # fit_once runs only without an ok status
    st = common.fit_once("nep", size, "nep.txt", [os.environ.get("ACEJAX_AC_NEP", "nep")], split_hash=h)
    if st["status"] == "ok":
        st.update(generations=done + run, fit_wall_s=st["fit_wall_s"] + prev.get("fit_wall_s", 0.0) * bool(done),
                  n_params=prev.get("n_params") or n_params((d / "fit.log").read_text()), e0_per_atom=E0_SI)
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
