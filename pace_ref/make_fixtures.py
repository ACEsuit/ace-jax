"""Random-coefficient .yace fixtures + python-ace E/F/stress references.

Run in the python-ace venv (pace_ref/.venv).  Coefficients are random: parity
between two evaluators does not need a physical potential, and random values
exercise every term.  Writes fixtures/pace/{name}.yace and {name}_ref.npz.
"""
import copy, pathlib, tempfile, zlib
import numpy as np, yaml
from ase import Atoms
from ase.build import bulk
from pyace import ACEBBasisSet, PyACECalculator, create_multispecies_basis_config

OUT = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "pace"


def emb(nd, npot, fs, rc=1e5, drc=250.0):
    return {"npot": npot, "fs_parameters": fs, "ndensity": nd,
            "rho_core_cut": rc, "drho_core_cut": drc}


def bond(radbase, **kw):
    b = {"radbase": radbase, "radparameters": [5.25], "rcut": 5.0, "dcut": 0.01,
         "NameOfCutoffFunction": "cos", "inner_cutoff_type": "density"}   # pyace defaults to "distance"
    b.update(kw)
    return b


FUNCS1 = {"number_of_functions_per_element": 40,
          "ALL": {"nradmax_by_orders": [8, 4, 3, 2], "lmax_by_orders": [0, 3, 2, 1]}}
FUNCS2 = {"number_of_functions_per_element": 60,
          "ALL": {"nradmax_by_orders": [8, 4, 3], "lmax_by_orders": [0, 3, 2]}}

CASES = {
    "si_chebexpcos": (["Si"], emb(1, "FinnisSinclairShiftedScaled", [1, 1]), {"ALL": bond("ChebExpCos")}, FUNCS1),
    "si_chebpow_fs": (["Si"], emb(2, "FinnisSinclair", [1, 1, 1, 0.5]), {"ALL": bond("ChebPow", radparameters=[2.0])}, FUNCS1),
    "si_cheblinear": (["Si"], emb(1, "FinnisSinclairShiftedScaled", [1, 1]), {"ALL": bond("ChebLinear")}, FUNCS1),
    "gesi_sbessel": (["Ge", "Si"], emb(2, "FinnisSinclairShiftedScaled", [1, 1, 1, 0.5]),
                     {"ALL": bond("SBessel"), ("Ge", "Ge"): bond("SBessel", rcut=4.2)},
                     {"number_of_functions_per_element": 60,
                      "ALL": {"nradmax_by_orders": [8, 4, 3], "lmax_by_orders": [0, 3, 2]},
                      "Ge": {"nradmax_by_orders": [6, 2, 2], "lmax_by_orders": [0, 2, 1]}}),
    "sige_density_core": (["Si", "Ge"], emb(1, "FinnisSinclairShiftedScaled", [1, 1], rc=50.0, drc=20.0),
                          {"ALL": bond("ChebExpCos", **{"core-repulsion": [100.0, 5.0]})}, FUNCS2),
    "sige_distance": (["Si", "Ge"], emb(1, "FinnisSinclairShiftedScaled", [1, 1]),
                      {"ALL": bond("ChebExpCos", inner_cutoff_type="distance", r_in=1.6, delta_in=0.4,
                                   **{"core-repulsion": [100.0, 5.0]})}, FUNCS2),
    "sige_zbl": (["Si", "Ge"], emb(1, "FinnisSinclairShiftedScaled", [1, 1]),
                 {"ALL": bond("ChebExpCos", inner_cutoff_type="zbl", r_in=1.6, delta_in=0.4,
                                   **{"core-repulsion": [1.0, 1.0]})}, FUNCS2),   # prehc = ZBL prefactor
}


def structures(elements):
    rng = np.random.default_rng(0)
    b = bulk("Si", "diamond", a=5.43, cubic=True).repeat(2)
    b.rattle(0.05, seed=0)
    if len(elements) > 1:
        b.numbers = np.random.default_rng(1).choice(
            [Atoms(e).numbers[0] for e in elements], len(b))
    else:
        b.numbers[:] = Atoms(elements[0]).numbers[0]
    z0 = b.numbers[0]
    dimer = Atoms(numbers=[z0, b.numbers[-1]], positions=[[0, 0, 0], [2.3, 0, 0]],
                  cell=np.eye(3) * 20, pbc=False)
    iso = Atoms(numbers=[z0], positions=[[0, 0, 0]], cell=np.eye(3) * 20, pbc=False)
    close = b.copy()
    d = close.get_distance(0, 1, mic=True, vector=True)
    close.positions[0] = close.positions[1] - 1.1 * d / np.linalg.norm(d)
    return {"bulk": b, "dimer_2.3": dimer, "isolated": iso, "close": close}


def evaluate(path, at):
    at = at.copy()
    at.calc = PyACECalculator(str(path))
    E = at.get_potential_energy()
    F = at.get_forces()
    S = at.get_stress() if at.pbc.all() else np.zeros(6)
    return E, F, S


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (els, embd, bonds, funcs) in CASES.items():
        cfg = {"deltaSplineBins": 0.001, "elements": els, "embeddings": {"ALL": embd},
               "bonds": bonds, "functions": funcs}
        bb = ACEBBasisSet(create_multispecies_basis_config(cfg))
        rng = np.random.default_rng(zlib.crc32(name.encode()))   # hash() is salted per process
        c = np.asarray(bb.all_coeffs)                          # API line 1
        bb.all_coeffs = rng.normal(scale=0.3, size=c.shape)     # API line 2 (a settable property)
        cb = bb.to_ACECTildeBasisSet()                          # API line 3
        yace = OUT / f"{name}.yace"
        cb.save_yaml(str(yace))
        tight = pathlib.Path(tempfile.mkdtemp()) / f"{name}.yace"
        tight.write_text(yace.read_text().replace("deltaSplineBins: 0.001", "deltaSplineBins: 0.0001"))
        assert "deltaSplineBins: 0.0001" in tight.read_text(), "deltaSplineBins key not found"
        ref = {"struct_names": np.asarray(list(structures(els)))}
        import pyace
        ref["pyace_version"] = np.asarray(getattr(pyace, "__version__", "unknown"))
        for s, at in structures(els).items():
            ref[f"Z_{s}"], ref[f"pos_{s}"] = at.numbers, at.positions
            ref[f"cell_{s}"], ref[f"pbc_{s}"] = at.cell.array, at.pbc
            for tag, p in (("tight", tight), ("ship", yace)):
                E, F, S = evaluate(p, at)
                ref[f"E_{tag}_{s}"], ref[f"F_{tag}_{s}"], ref[f"S_{tag}_{s}"] = E, F, S
        np.savez(OUT / f"{name}_ref.npz", **ref)
        print(f"{name}: {yace.stat().st_size // 1024} kB")


if __name__ == "__main__":
    main()
