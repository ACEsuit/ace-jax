"""Label the MLIP-school-derived tutorials' structures with MIT-licensed MACE models.

Needs mace-torch (and CPU torch) -- an environment of its own, not ace-jax's:

    uv venv mace-env && uv pip install --python mace-env/bin/python \\
        --index-url https://download.pytorch.org/whl/cpu torch
    uv pip install --python mace-env/bin/python mace-torch ase
    PYTHONPATH=src mace-env/bin/python docs/user/tutorials/data/school/make_labels.py a0
    PYTHONPATH=src mace-env/bin/python docs/user/tutorials/data/school/make_labels.py all \\
        --e1x-source <MLIP-school-2026>/notebooks/reference/e1x-bulk-reference.xyz

`a0` prints MACE-MPA-0's diamond-Si lattice constant (set it as
ace_jax.tutorials.structures.E1_A0 before labelling, since the vacancy pair is
built at it). `all` writes, next to this script:

    e1/labels-mpa-0.xyz          E1's slider grid (30 settings x 10 cells) and vacancy pair
    e1x/reference.xyz            the school's 250 E1x frames, relabelled
    c/labels-mpa-0.xyz, c/labels-mp-0b3.xyz   C's reference pair and surface recipe

Labels are float64 on CPU; see README.md for the models and their licence."""
import argparse
import pathlib
import time

import numpy as np

from ace_jax.tutorials import labels as L
from ace_jax.tutorials import structures as T

HERE = pathlib.Path(__file__).parent


def lattice_constant(model="mpa-0"):
    from ase.build import bulk
    from ase.eos import EquationOfState
    calc = L._mace_calculator(model)
    a = np.linspace(5.30, 5.60, 13)
    E = []
    for x in a:
        at = bulk("Si", "diamond", a=x, cubic=True)
        at.calc = calc
        E.append(at.get_potential_energy())
    v0, _, _ = EquationOfState(a ** 3, np.asarray(E), eos="birchmurnaghan").fit()
    return float(v0 ** (1 / 3))


def _write(path, structures, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    lab = L.label(structures, model=model, calculator=L._mace_calculator(model))
    L.write_cache(path, lab)
    print(f"{path.relative_to(HERE)}: {len(lab)} structures, {model}, {time.time() - t:.0f} s", flush=True)


def e1x_structures(source):
    from ase import Atoms
    from ace_jax.fit.xyz import read_extxyz
    out = []
    for f in read_extxyz(source):                 # structures only: the old labels are dropped
        a = Atoms(numbers=f.numbers, positions=f.positions, cell=f.cell, pbc=f.pbc)
        a.info["config_type"] = str(f.info.get("config_type", "bulk"))
        out.append(a)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["a0", "all", "e1", "e1x", "c"])
    p.add_argument("--e1x-source", default=None, help="the school's e1x-bulk-reference.xyz")
    a = p.parse_args()
    if a.what == "a0":
        print(f"MACE-MPA-0 diamond Si a0 = {lattice_constant():.4f} A (structures.E1_A0 = {T.E1_A0})")
        return
    if abs(lattice_constant() - T.E1_A0) > 5e-4:
        raise SystemExit("structures.E1_A0 is not MPA-0's lattice constant: run `a0` and update it first")
    if a.what in ("all", "e1"):
        grid = [c for s in T.E1_STRAINS for r in T.E1_RATTLES for c in T.e1_cells(s, r)]
        _write(HERE / "e1" / "labels-mpa-0.xyz", grid + list(T.e1_vacancy_pair()), "mpa-0")
    if a.what in ("all", "e1x"):
        if not a.e1x_source:
            raise SystemExit("--e1x-source is needed for e1x")
        _write(HERE / "e1x" / "reference.xyz", e1x_structures(a.e1x_source), "mpa-0")
    if a.what in ("all", "c"):
        c = T.c_structures()
        for m in ("mpa-0", "mp-0b3"):
            _write(HERE / "c" / f"labels-{m}.xyz", [c["bulk"], c["slab111"], *c["recipe"]], m)


if __name__ == "__main__":
    main()
