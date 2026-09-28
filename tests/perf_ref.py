"""Write frozen E/F/stress references from the CURRENT code (run once, before
any optimisation): tests/test_perf_parity.py holds every later change to them.

    uv run python tests/perf_ref.py
"""
import pathlib

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase import Atoms
from ase.build import bulk

from ace_jax.calc.point import ACECalculator

ROOT = pathlib.Path(__file__).parent.parent
OUT = pathlib.Path(__file__).parent / "fixtures" / "perf_ref"
PACE = sorted((ROOT / "fixtures" / "pace").glob("*.yace"))
ACE = [ROOT / "fixtures" / "si_ace_model.npz", ROOT / "fixtures" / "sige_nofit.npz"]


def cells(elements):
    """Periodic (triclinic, a small cell where rcut+skin spans several images)
    and an open cluster, with every element of the model present."""
    z = [int(e) for e in elements]
    rng = np.random.default_rng(0)
    per = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    per.set_cell(per.cell.array @ np.array([[1, 0.08, 0], [0, 1, 0.05], [0, 0, 1]]),
                 scale_atoms=True)                            # triclinic
    per.numbers = np.asarray(z)[rng.integers(0, len(z), len(per))]
    per.positions += rng.normal(0, 0.05, per.positions.shape)
    small = bulk("Si", "diamond", a=5.43)                      # 2 atoms: many images per pair
    small.numbers = np.asarray(z)[np.arange(len(small)) % len(z)]
    clus = per.copy()
    clus.pbc = False
    clus.center(vacuum=6.0)
    iso = Atoms(numbers=[z[0], z[-1]], positions=[[0, 0, 0], [20.0, 0, 0]],
                cell=[40, 40, 40], pbc=False)                # atoms with no neighbours
    return {"tric": per, "small": small, "cluster": clus, "isolated": iso}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from ace_jax.eval import load
    for path in PACE + ACE:
        _, meta, _ = load(str(path))
        for cname, at in cells(meta["elements"]).items():
            for layout in ("sparse", "dense"):
                for dt in ("float64", "float32"):
                    calc = ACECalculator(str(path), layout=layout, dtype=getattr(jnp, dt))
                    a = at.copy()
                    a.calc = calc
                    E = a.get_potential_energy()
                    F = a.get_forces()
                    S = a.get_stress() if a.pbc.all() else np.zeros(6)
                    np.savez(OUT / f"{path.stem}_{cname}_{layout}_{dt}.npz", E=E, F=F, S=S,
                             numbers=a.numbers, positions=a.positions, cell=a.cell.array,
                             pbc=a.pbc)
    print("wrote", len(list(OUT.glob("*.npz"))), "references to", OUT)


if __name__ == "__main__":
    main()
