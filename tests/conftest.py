"""Both model families are exercised by every test, so neither branch can rot.

  si_fitted.npz     ace1_model : splined rbasis, splined pair, SPHERICAL Ylm,
                                 ACE1_PolyEnvelope1sR
  si_ace_model.npz  ace_model  : ANALYTIC rbasis (live Wnlq + 3-term recursion),
                                 splined pair, SOLID Ylm, PolyEnvelope1sR

Note the pair basis is splined in both: `ace_model` splinifies it
(ace_heuristics.jl:213), so the radial branches are per-basis, not per-model.
"""
import os

# Two host CPU devices so tests/test_gp_sharding.py runs in the full suite.
# Must be set before ANY jax import (the CPU backend fixes its device count at
# initialisation), so this module imports nothing from jax at top level.
_flag = "--xla_force_host_platform_device_count=2"
if _flag not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _flag).strip()

import pathlib

import pytest

ROOT = pathlib.Path(__file__).parent.parent

# ACEJAX_FIXTURE_DIR points the whole suite at a different set of exports -- used
# by the divergence job in CI, which regenerates them from the Julia in the
# working tree so the reference values are fresh rather than committed.
FIXTURE_DIR = pathlib.Path(os.environ.get("ACEJAX_FIXTURE_DIR", ROOT / "fixtures"))

# Missing fixtures normally skip, so a developer without Julia can still run the
# suite.  In CI that would be a silent pass: a regeneration that produced nothing
# would look identical to a green run.  ACEJAX_REQUIRE_FIXTURES turns the skip
# into a failure.
REQUIRE = bool(os.environ.get("ACEJAX_REQUIRE_FIXTURES"))

MODELS = {
    "ace1_spline_spherical": "si_fitted.npz",
    "ace_analytic_solid": "si_ace_model.npz",
    # two species, random (unfitted) weights: exercises every per-species index
    # path -- Wnlq[:,:,iz,jz], E0[z], WB[:,z] and the folded ctilde[:,z]
    "ace1_two_species": "sige_nofit.npz",
}


def pytest_generate_tests(metafunc):
    if "npz" in metafunc.fixturenames:
        ids, paths = [], []
        for name, fname in MODELS.items():
            p = FIXTURE_DIR / fname
            ids.append(name)
            if p.exists():
                paths.append(pytest.param(p))
            elif REQUIRE:
                # Abort collection outright.  An xfail or a skip would still be a
                # green run, which is the failure mode this guard exists to stop.
                raise FileNotFoundError(
                    f"ACEJAX_REQUIRE_FIXTURES is set but {p} is missing -- the "
                    f"Julia export did not produce it")
            else:
                paths.append(pytest.param(p, marks=pytest.mark.skip(
                    reason=f"missing fixture {p.name}; see julia/export_model.jl")))
        metafunc.parametrize("npz", paths, ids=ids)


def species_index(z):
    """Map the exported per-atom atomic numbers onto model species indices.

    `node_z` is an index into the model's `elements` (i2z) table, NOT an atomic
    number. Tests used to hardcode `zeros(n_nodes)`, which is correct only for a
    single-species model and silently wrong for any other -- so a two-element
    export could not be tested at all.
    """
    import jax.numpy as jnp
    import numpy as np
    i2z = list(np.asarray(z["elements"]).ravel())
    lookup = {int(zz): i for i, zz in enumerate(i2z)}
    return jnp.asarray([lookup[int(a)] for a in np.asarray(z["test_Z"]).ravel()],
                       dtype=jnp.int32)
