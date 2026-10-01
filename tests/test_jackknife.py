import numpy as np
from ase.build import bulk


def _cfg(at):
    from ace_jax.fit.data import Config
    return Config(at.get_positions(), at.get_atomic_numbers(), at.get_cell().array, at.get_pbc(),
                  0.0, np.zeros((len(at), 3)), np.zeros((3, 3)), 1.0, 1.0, 1.0)


def test_config_blocks_small_and_large():
    from ace_jax.fit.clusters import config_blocks
    ell = 3 * 6.25
    assert config_blocks(_cfg(bulk("Ni", "fcc", a=3.65, cubic=True).repeat(4)), ell) is None   # 14.6 A
    big = bulk("Ni", "fcc", a=3.65, cubic=True).repeat((12, 4, 4))                              # 43.8 A in x
    b = config_blocks(_cfg(big), ell)
    frac = big.get_scaled_positions(wrap=True)[:, 0]
    assert len(np.unique(b)) == 2                                                               # floor(43.8/18.75)
    assert len({(int(fx * 2), int(bi)) for fx, bi in zip(frac, b)}) == 2                        # id = f(floor(2 fx))


def test_config_blocks_nonperiodic_uses_bounding_box():
    from ace_jax.fit.clusters import config_blocks
    at = bulk("Ni", "fcc", a=3.65, cubic=True).repeat((16, 16, 2))                              # 58.4 A in x, y
    at.pbc = (False, False, True)
    at.center(vacuum=10.0, axis=(0, 1))
    assert len(np.unique(config_blocks(_cfg(at), 3 * 6.25))) == 9                               # 3 x 3 x 1


def test_row_clusters_ids():
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.data import build_dataset
    small = _cfg(bulk("Ni", "fcc", a=3.65, cubic=True).repeat(2))
    large = _cfg(bulk("Ni", "fcc", a=3.65, cubic=True).repeat((12, 4, 4)))
    ds = build_dataset([small, large], {"elements": [28], "rcut": 6.25}, np.zeros(1), 2)
    rc, K = row_clusters(ds, [small, large], 3 * 6.25)
    b, ns, nl = rc[0], len(small.numbers), len(large.numbers)
    assert b["E"][0] == b["V"][0] == 0 and np.all(b["F"][:ns] == 0)                            # small: one id
    fl = b["F"][ns:ns + nl]
    assert set(fl.tolist()) == {1, 2} and b["E"][1] == b["V"][1] == 3 and K == 4               # blocks, then E/V
    assert np.all(b["F"][ns + nl:] == -1)
    rc_inf, K_inf = row_clusters(ds, None, float("inf"))
    assert K_inf == 2 and np.all(rc_inf[0]["F"][ns:ns + nl] == 1)


def test_config_blocks_degenerate_cell_uses_cartesian_bounding_box():
    from ace_jax.fit.clusters import config_blocks
    for pbc in (False, True):
        at = bulk("Ni", "fcc", a=3.65, cubic=True).repeat((12, 4, 4))
        at.set_cell(np.zeros((3, 3)))
        at.pbc = pbc
        b = config_blocks(_cfg(at), 3 * 6.25)
        assert b is not None and len(np.unique(b)) == 2
        x = at.get_positions()[:, 0]
        assert len({(int(xi > x.min() + (x.max() - x.min()) / 2), int(bi)) for xi, bi in zip(x, b)}) == 2


def test_row_clusters_validates_configs():
    import pytest
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.data import build_dataset
    small = _cfg(bulk("Ni", "fcc", a=3.65, cubic=True).repeat(2))
    ds = build_dataset([small], {"elements": [28], "rcut": 6.25}, np.zeros(1), 1)
    with pytest.raises(ValueError):
        row_clusters(ds, None, 18.75)
    with pytest.raises(ValueError):
        row_clusters(ds, [small, small], 18.75)
