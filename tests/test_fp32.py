# tests/test_fp32.py
"""float32 evaluation under x64 (docs/dev/fp32-results.md): the calculator and the lammps-jax
export cast the evaluation model to the requested dtype (a float64 model traced with float32
inputs promotes back to float64), float32 agrees with float64 to the model's own float32 error,
and `precision_check` records and flags that error."""
import re

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from ace_jax.calc import ACECalculator
from ace_jax.eval import load, with_dtype
from conftest import FIXTURE_DIR, require_optional

SI = FIXTURE_DIR / "si_fitted.npz"


def _cell(rattle=0.05, seed=3):
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    at.rattle(rattle, seed=seed)
    return at


def _floating_dtypes(tree):
    return {str(x.dtype) for x in jax.tree.leaves(tree) if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.floating)}


def test_with_dtype_casts_floating_leaves_only():
    model, _, _ = load(SI)
    m32 = with_dtype(model, np.float32)
    assert _floating_dtypes(m32) == {"float32"}
    ints = lambda t: [x for x in jax.tree.leaves(t) if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.integer)]
    assert [x.dtype for x in ints(m32)] == [x.dtype for x in ints(model)]


@pytest.mark.parametrize("preloaded", [False, True])
def test_calculator_float32_evaluates_in_float32(preloaded):
    """A path and a pre-loaded float64 model alike: the evaluation model is float32 (the full
    model is kept as given), and E/F/stress agree with float64 to the fixture's float32 error
    (measured 1.7e-3 meV/atom, 6.3e-6 force, 1.6e-5 stress)."""
    model, meta, _ = load(SI)
    args = (model, meta) if preloaded else (SI,)
    c32, c64 = ACECalculator(*args, dtype=np.float32), ACECalculator(*args)
    assert _floating_dtypes(c32.eval_model) == {"float32"}
    assert _floating_dtypes(c64.eval_model) == {"float64"}
    a32, a64 = _cell(), _cell()
    a32.calc, a64.calc = c32, c64
    F32, F64 = a32.get_forces(), a64.get_forces()
    assert abs(a32.get_potential_energy() - a64.get_potential_energy()) / len(a32) < 2e-5        # eV/atom
    assert np.linalg.norm(F32 - F64) / np.linalg.norm(F64) < 5e-5
    S32, S64 = a32.get_stress(), a64.get_stress()
    assert np.linalg.norm(S32 - S64) / np.linalg.norm(S64) < 1e-4


def test_float32_nve_drift_matches_float64():
    """500 fs of NVE at 600 K: float32 drifts no more than float64 does plus a margin (both
    ~0.02 meV/atom on the fixture), so the float32 forces stay conservative."""
    drift = {}
    for name, dt in (("f64", None), ("f32", np.float32)):
        at = _cell()
        at.calc = ACECalculator(SI, dtype=dt)
        MaxwellBoltzmannDistribution(at, temperature_K=600, rng=np.random.default_rng(0))
        md = VelocityVerlet(at, 1.0 * units.fs)
        e0 = at.get_potential_energy() + at.get_kinetic_energy()
        es = []
        for _ in range(10):
            md.run(50)
            es.append(at.get_potential_energy() + at.get_kinetic_energy() - e0)
        drift[name] = np.abs(es).max() / len(at)
    assert drift["f32"] < drift["f64"] + 2e-5, drift


def test_precision_check_fields_and_warning():
    from ace_jax.calc.precision import PRECISION_WARN, precision_check, warn_precision
    model, meta, _ = load(SI)
    chk = precision_check(model, meta, _cell())
    assert chk["dtype"] == "float32" and chk["n_atoms"] == 64
    assert chk["dF_rel"] < PRECISION_WARN["dF_rel"] and chk["dE_meV_atom"] < PRECISION_WARN["dE_meV_atom"]
    assert warn_precision(chk) == []
    with pytest.warns(UserWarning, match="loses accuracy in float32"):
        assert warn_precision({**chk, "dF_rel": 1e-2}) == ["dF_rel"]


@pytest.mark.parametrize("layout", ["dense", "sparse"])
def test_float32_bundle_lowers_without_float64(tmp_path, monkeypatch, layout):
    """Under x64, as `aj export` runs: before the cast a float32 bundle of a float64 model
    lowered to ~400 f64 ops with only its I/O in f32.  Now no f64 tensor remains (the dense
    total energy's compensated scalar is f64 by design, a rank-0 tensor<f64>)."""
    require_optional("lammps_jax")
    import jax.export as je
    from ace_jax.export.lammps import export_lammps, neighbour_capacity
    texts, orig = [], je.export

    def spy(fn, **kw):
        ex = orig(fn, **kw)

        def call(*a):
            e = ex(*a)
            texts.append(e.mlir_module())
            return e
        return call
    monkeypatch.setattr(je, "export", spy)
    model, meta, _ = load(SI)
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3))
    cap = neighbour_capacity(at, float(meta["rcut"]))
    b = export_lammps(model, meta, tmp_path / "b.json", max_atoms=cap["max_atoms"], max_edges=cap["max_edges"],
                      k_dense=cap["k_dense"], max_owned=cap["max_owned"], dtype="float32", layout=layout,
                      precision_structure=_cell())
    assert texts and all(not re.search(r"x f64>|xf64>", t) for t in texts)
    assert all(re.search(r"xf32>", t) for t in texts)
    chk = b["ace_jax"]["precision_check"]
    assert chk["dtype"] == "float32" and chk["dF_rel"] < 1e-4
