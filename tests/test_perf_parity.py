"""Every optimisation keeps E, F and stress equal to the frozen references
(tests/fixtures/perf_ref, written by tests/perf_ref.py from the code before the
speed-ups): 1e-12 relative in float64. float32 is gated on accuracy against
the float64 reference: an absolute bound set by float32 precision, not
agreement with the old float32 rounding (docs/dev/perf-optimisation-spec.md).
float32 noise moves with summation order (neighbour order: ASE vs
matscipy-neighbours; numpy/XLA build: macOS arm64 vs Linux x86), so a gate
relative to one sampled old-float32 error rejects a reordering, not a bug.
See F32_TOL below for the bounds and the measurements behind them."""
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


# float32 accuracy against the float64 reference, per quantity: |x32 - x64|
# (max-abs over components) <= rtol * max|x64| + atol.  rtol covers the
# relative rounding of a large result; atol the cancellation noise of a small
# (or zero, e.g. a perfect-crystal force) one, whose float32 error scales with
# the per-bond contributions that cancel, not with the net value.  eps32 = 1.2e-7.
#   E: rtol 5e-6 (~40 eps32), atol 1e-6 eV per atom
#   F: rtol 2e-5,            atol 1e-3 eV/A
#   S: rtol 5e-5,            atol 1e-5 eV/A^3
# Measured over all 144 float32 cases (72 references x skin 0/1), as the
# worst error / bound ratio, in ASE neighbour order | matscipy-neighbours order
# (macOS arm64):  E 0.287 | 0.251,  F 0.210 | 0.244,  S 0.174 | 0.219.
# Worst components: E 1.7e-6 relative (large E), 2.7e-7 eV/atom (small E);
# F 5.4e-6 relative (max|F64| up to 3.1e3 eV/A), 2.4e-4 eV/A absolute on a
# zero-force cell; S 1.8e-5 relative.  CI's Linux x86 drew 3.05e-4 eV/A on the
# zero-force si_chebpow_fs small cell (ratio 0.305): every bound keeps >= 3x.
F32_TOL = {"E": (5e-6, 1e-6), "F": (2e-5, 1e-3), "S": (5e-5, 1e-5)}


def _f32_ok(x, x64, kind, n_atoms=1):
    """float32 x within F32_TOL[kind] of the float64 reference x64 (the energy's
    atol is per atom)."""
    rtol, atol = F32_TOL[kind]
    err = np.max(np.abs(np.asarray(x, np.float64) - x64))
    bound = rtol * np.max(np.abs(x64)) + atol * (n_atoms if kind == "E" else 1)
    assert err <= bound, f"{kind}: float32 error {err:.3g} > bound {bound:.3g}"


# the sparse layout never takes the skin path (ACECalculator._calculate: skin > 0 and
# layout != "sparse"), so sparse x skin=1 would repeat its skin=0 case exactly
SKIN_CASES = [pytest.param(ref, skin, id=f"{skin}-{ref.stem}") for ref in CASES
              for skin in ((0.0,) if ref.stem.rsplit("_", 3)[2] == "sparse" else (0.0, 1.0))]


@pytest.mark.parametrize("ref, skin", SKIN_CASES)
def test_matches_frozen_reference(ref, skin):
    stem, cname, layout, dt = ref.stem.rsplit("_", 3)
    r = np.load(ref)
    at = Atoms(numbers=r["numbers"], positions=r["positions"], cell=r["cell"], pbc=r["pbc"])
    at.calc = ACECalculator(str(_model_path(stem)), layout=layout, dtype=getattr(jnp, dt), skin=skin)
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
    _f32_ok(E, float(r64["E"]), "E", len(at))
    _f32_ok(F, r64["F"], "F")
    if S is not None:
        _f32_ok(S, r64["S"], "S")
