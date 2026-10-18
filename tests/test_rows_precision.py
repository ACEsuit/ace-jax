# tests/test_rows_precision.py
"""The fit's design rows are traced at the highest matmul precision whatever the caller's
default: on Ampere+ GPUs JAX's default float32 matmul is TF32, which corrupted float32 force
rows ~250x (docs/dev/fp32-results.md).  Checked on the jaxpr, so it runs on the CPU: every
dot_general in the rows (the model's Jacobian included) must carry Precision.HIGHEST."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from jax.extend.core import ClosedJaxpr, Jaxpr

from conftest import FIXTURE_DIR

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")


def _dot_precisions(jaxpr, out):
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            out.append(eqn.params["precision"])
        for v in eqn.params.values():
            for sub in (v if isinstance(v, (list, tuple)) else [v]):
                if isinstance(sub, ClosedJaxpr):
                    _dot_precisions(sub.jaxpr, out)
                elif isinstance(sub, Jaxpr):
                    _dot_precisions(sub, out)
    return out


@pytest.fixture(scope="module")
def problem():
    from ace_jax.eval import load
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    cfgs = [c for c in load_configs(XYZ, "dft_energy", "dft_force", "dft_virial") if len(c.numbers) < 64][:4]
    ds = build_dataset(cfgs, meta, np.asarray(z["E0"]), 2)
    g = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                 NZ=len(meta["elements"]), C=2)
    return model, g, jax.tree.map(lambda a: a[0], ds)


@pytest.mark.parametrize("entry", ["linear_rows", "linear_rows_bounded", "linear_rows_chunked"])
def test_rows_dot_generals_are_highest(problem, entry):
    import ace_jax.fit.rows as rows
    model, g, batch = problem
    fn = getattr(rows, entry)
    with jax.default_matmul_precision("bfloat16"):                  # a caller's low default
        jaxpr = jax.make_jaxpr(lambda b: fn(model, g, b))(batch)
    precs = _dot_precisions(jaxpr.jaxpr, [])
    assert precs, "no dot_general traced"
    hi = jax.lax.Precision.HIGHEST
    assert all(p is not None and all(q == hi for q in (p if isinstance(p, tuple) else (p,))) for p in precs), \
        set(map(str, precs))
