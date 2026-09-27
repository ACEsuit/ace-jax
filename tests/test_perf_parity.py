"""Every optimisation keeps E, F and stress equal to the frozen references
(tests/fixtures/perf_ref, written by tests/perf_ref.py from the code before the
speed-ups): 1e-12 relative in float64, 1e-5 in float32."""
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase import Atoms

from ace_jax.calc.point import ACECalculator

ROOT = pathlib.Path(__file__).parent.parent
REF = pathlib.Path(__file__).parent / "fixtures" / "perf_ref"


def _model_path(stem):
    p = ROOT / "fixtures" / "pace" / f"{stem}.yace"
    return p if p.exists() else ROOT / "fixtures" / f"{stem}.npz"


CASES = sorted(REF.glob("*.npz"))


@pytest.mark.parametrize("skin", [0.0, 1.0])
@pytest.mark.parametrize("ref", CASES, ids=[c.stem for c in CASES])
def test_matches_frozen_reference(ref, skin):
    stem, cname, layout, dt = ref.stem.rsplit("_", 3)
    r = np.load(ref)
    at = Atoms(numbers=r["numbers"], positions=r["positions"], cell=r["cell"], pbc=r["pbc"])
    kw = {} if "skin" not in ACECalculator.__init__.__code__.co_varnames else {"skin": skin}
    at.calc = ACECalculator(str(_model_path(stem)), layout=layout, dtype=getattr(jnp, dt), **kw)
    tol = 1e-12 if dt == "float64" else 1e-5
    scale = max(1.0, abs(float(r["E"])))
    assert abs(at.get_potential_energy() - float(r["E"])) <= tol * scale
    np.testing.assert_allclose(at.get_forces(), r["F"], rtol=tol, atol=tol * scale / len(at))
    if at.pbc.all():
        np.testing.assert_allclose(at.get_stress(), r["S"], rtol=tol, atol=tol * scale)
