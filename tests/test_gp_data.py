import pathlib

import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import load
from ace_jax.fit.data import VOIGT, build_dataset, load_configs

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
NPZ = FIXTURE_DIR / "si_fitted.npz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")


@pytest.fixture(scope="module")
def configs():
    return load_configs(XYZ, energy_key="dft_energy", force_key="dft_force",
                        virial_key="dft_virial")


def test_load_configs_reads_keys_and_structural_weights(configs):
    # configs[0] is the isolated atom: it carries dft_energy and dft_force but no
    # dft_virial (ACEfit's design matrix has 1052 = 53*7 + 3*229 - 6 rows for
    # the same reason), so the virial checks use configs[1].
    assert configs[0].virial is None and configs[0].forces.shape == (1, 3)
    c = configs[1]
    assert c.forces.shape == (len(c.numbers), 3) and c.virial.shape == (3, 3)
    assert abs(c.w_E - 1.0 / np.sqrt(len(c.numbers))) < 1e-15 and c.w_F == 1.0


def test_dataset_shapes_and_padding(configs):
    model, meta, z = load(NPZ)
    ds = build_dataset(configs[:7], meta, np.asarray(z["E0"]), configs_per_batch=3)
    nb, C = ds.y_E.shape
    assert nb == 3 and C == 3
    assert ds.node_cfg.shape == ds.node_mask.shape == ds.node_z.shape
    assert ds.node_z.shape[1] % 32 == 0
    # last batch holds one config; its other two slots are masked with zero weight
    assert ds.cfg_mask[2].tolist() == [True, False, False]
    assert float(ds.w_E[2, 1]) == 0.0 and float(ds.w_V[2, 1]) == 0.0
    # padded nodes point at the garbage slot C and carry zero force weight
    pad = ~ds.node_mask
    assert bool(jnp.all(ds.node_cfg[pad] == C)) and float(ds.w_F[pad].max()) == 0.0
    # every live neighbour slot points at a live node of the same config
    live = ds.nbr_mask
    nc = jnp.take_along_axis(ds.node_cfg, ds.nbr.reshape(nb, -1), 1).reshape(ds.nbr.shape)
    assert bool(jnp.all((nc == ds.node_cfg[:, :, None])[live]))
    assert ds.rij.shape[:3] == ds.nbr.shape


def test_energy_target_subtracts_e0_and_virial_is_voigt(configs):
    model, meta, z = load(NPZ)
    E0 = np.asarray(z["E0"])
    ds = build_dataset(configs[:2], meta, E0, configs_per_batch=2)
    for k, c in enumerate(configs[:2]):
        z_idx = np.array([list(meta["elements"]).index(int(a)) for a in c.numbers])
        assert abs(float(ds.y_E[0, k]) - (c.energy - E0[z_idx].sum())) < 1e-10
        assert float(ds.n_atoms[0, k]) == len(c.numbers)
    # configs[1] has a virial; configs[0] (isolated atom) does not, and an
    # absent observation must carry ZERO weight
    c = configs[1]
    assert np.allclose(np.asarray(ds.y_V[0, 1]), [c.virial[i, j] for i, j in VOIGT])
    assert float(ds.w_V[0, 1]) == c.w_V and float(ds.w_V[0, 0]) == 0.0


def test_weights_without_default_raises():
    with pytest.raises(ValueError, match="'default' entry"):
        load_configs(XYZ, energy_key="dft_energy", weights={"bulk": {"E": 1.0, "F": 1.0, "V": 1.0}})
