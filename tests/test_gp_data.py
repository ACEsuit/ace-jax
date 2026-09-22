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


def test_default_factors_reproduce_classic_weights_and_type_idx(tmp_path):
    """Backwards-compat (Task 13): factors=None must reproduce the classic
    weights-dict computation EXACTLY -- structural 1/sqrt(n) on E,V (1 on F),
    times the resolved per-config-type {E,F,V} dict -- and Task 6's type_idx
    (default type -> 0, named types -> 1, 2, ... in insertion order) must
    still come out unchanged."""
    from ase import Atoms
    from ase.io import write

    a_bulk = Atoms("Si4", positions=np.zeros((4, 3)), cell=[5.0] * 3, pbc=True)
    a_bulk.info["dft_energy"] = -20.0
    a_bulk.arrays["dft_force"] = np.zeros((4, 3))
    a_bulk.info["config_type"] = "bulk"          # not a named weights entry -> default

    a_def = Atoms("Si9", positions=np.zeros((9, 3)), cell=[6.0] * 3, pbc=True)
    a_def.info["dft_energy"] = -40.0
    a_def.arrays["dft_force"] = np.zeros((9, 3))
    a_def.info["config_type"] = "defect"          # named weights entry

    p = tmp_path / "two_types.xyz"
    write(p, [a_bulk, a_def])

    weights = {"default": {"E": 1.0, "F": 1.0, "V": 1.0},
               "defect": {"E": 10.0, "F": 5.0, "V": 2.0}}
    cfgs = load_configs(p, energy_key="dft_energy", force_key="dft_force", weights=weights)

    assert cfgs[0].type_idx == 0
    assert abs(cfgs[0].w_E - 1.0 / np.sqrt(4)) < 1e-12
    assert cfgs[0].w_F == 1.0
    assert abs(cfgs[0].w_V - 1.0 / np.sqrt(4)) < 1e-12

    assert cfgs[1].type_idx == 1
    assert abs(cfgs[1].w_E - 10.0 / np.sqrt(9)) < 1e-12
    assert cfgs[1].w_F == 5.0
    assert abs(cfgs[1].w_V - 2.0 / np.sqrt(9)) < 1e-12


def test_perconfig_factor_changes_only_that_configs_weight(tmp_path):
    """A PerConfig factor (Task 12) composed with Structural must change only
    the config that carries the metadata key -- proves `factors` is actually
    wired into load_configs, not just accepted and ignored."""
    from ase import Atoms
    from ase.io import write

    from ace_jax.fit.weights import PerConfig, Structural

    a1 = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[5.0] * 3, pbc=True)
    a1.info["dft_energy"] = -10.0
    a1.arrays["dft_force"] = np.zeros((2, 3))
    a1.info["w"] = 3.0

    a2 = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[5.0] * 3, pbc=True)
    a2.info["dft_energy"] = -10.0
    a2.arrays["dft_force"] = np.zeros((2, 3))
    # a2 carries no "w" key -> PerConfig is neutral (1.0)

    p = tmp_path / "perconfig.xyz"
    write(p, [a1, a2])

    cfgs = load_configs(p, energy_key="dft_energy", force_key="dft_force",
                        factors=[Structural(), PerConfig(key="w")])
    assert abs(cfgs[0].w_E - 3.0 / np.sqrt(2)) < 1e-12
    assert abs(cfgs[1].w_E - 1.0 / np.sqrt(2)) < 1e-12
    assert cfgs[0].w_E != cfgs[1].w_E
