"""lammps-jax resolves FFI custom calls at run time; a PACE model must lower to
stock StableHLO so it needs none."""
import pathlib
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import load

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"


@pytest.mark.parametrize("name", ["si_chebpow_fs", "sige_zbl", "gesi_sbessel"])
def test_exports_without_custom_calls(name):
    p = FIX / f"{name}.yace"
    if not p.exists():
        pytest.skip("fixtures not generated")
    model, _, _ = load(str(p))
    n, E = 8, 40
    f = jax.jit(lambda pos, z, s, r, m: jnp.sum(model.energy_from_positions(pos, z, s, r, m)))
    args = (jnp.zeros((n, 3)), jnp.zeros(n, jnp.int32), jnp.zeros(E, jnp.int32),
            jnp.zeros(E, jnp.int32), jnp.zeros(E, bool))
    text = jax.export.export(f)(*args).mlir_module()
    # match the op, not the substring: MLIR loc metadata carries this test's name
    assert "stablehlo.custom_call" not in text and "mhlo.custom_call" not in text
