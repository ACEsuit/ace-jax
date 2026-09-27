"""Every optimisation keeps E, F and stress equal to the frozen references
(tests/fixtures/perf_ref, written by tests/perf_ref.py from the code before the
speed-ups): 1e-12 relative in float64. float32 is gated on accuracy against
the float64 reference, not on bit-level agreement with the old float32
rounding (Task 2 ruling, docs/perf-optimisation-spec.md): two independently
rounded float32 results differ from each other by about their own error, so a
tight new-vs-old float32 gate would reject even a more accurate rewrite. See
_float32_gate below for the exact rule."""
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


def _err(x, x64):
    """Max-abs deviation from the float64 reference, relative to its scale."""
    return np.max(np.abs(x - x64)) / max(1.0, np.max(np.abs(x64)))


def _float32_gate(old32, x64):
    """Task 2 ruling: a float32 result must be no worse than the old float32
    result was, i.e. at most max(1e-5, 1.25 x old-float32 error) relative to
    the float64 reference. Plain bit-similarity to the old float32 rounding
    (a flat 1e-5 new-vs-old gate) is not used, because two independently
    rounded float32 evaluations of the same quantity differ from each other by
    about their own error against the true (float64) answer, which would
    reject a rewrite that is more accurate than the old one."""
    return max(1e-5, 1.25 * _err(old32, x64))


@pytest.mark.parametrize("skin", [0.0, 1.0])
@pytest.mark.parametrize("ref", CASES, ids=[c.stem for c in CASES])
def test_matches_frozen_reference(ref, skin):
    stem, cname, layout, dt = ref.stem.rsplit("_", 3)
    r = np.load(ref)
    at = Atoms(numbers=r["numbers"], positions=r["positions"], cell=r["cell"], pbc=r["pbc"])
    kw = {} if "skin" not in ACECalculator.__init__.__code__.co_varnames else {"skin": skin}
    at.calc = ACECalculator(str(_model_path(stem)), layout=layout, dtype=getattr(jnp, dt), **kw)
    E, F = at.get_potential_energy(), at.get_forces()
    S = at.get_stress() if at.pbc.all() else None

    if dt == "float64":
        tol = 1e-12
        scale = max(1.0, abs(float(r["E"])))
        assert abs(E - float(r["E"])) <= tol * scale
        np.testing.assert_allclose(F, r["F"], rtol=tol, atol=tol * scale / len(at))
        if S is not None:
            np.testing.assert_allclose(S, r["S"], rtol=tol, atol=tol * scale)
        return

    r64 = np.load(REF / f"{stem}_{cname}_{layout}_float64.npz")
    E64, F64, S64 = float(r64["E"]), r64["F"], r64["S"]
    E_old32, F_old32, S_old32 = float(r["E"]), r["F"], r["S"]

    assert _err(E, E64) <= _float32_gate(E_old32, E64)
    assert _err(F, F64) <= _float32_gate(F_old32, F64)
    if S is not None:
        assert _err(S, S64) <= _float32_gate(S_old32, S64)
