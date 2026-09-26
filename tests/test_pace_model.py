import pathlib
import jax
import numpy as np
import pytest
from conftest import pace_fixture

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase import Atoms

from ace_jax.eval import ACECalculator, load, sparse_graph
from ace_jax.eval.pace_model import PACEModel, load_yace

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"
NAMES = ["si_chebexpcos", "si_chebpow_fs", "si_cheblinear", "gesi_sbessel",
         "sige_density_core", "sige_distance", "sige_zbl"]


def _fixture(name):
    y, r = FIX / f"{name}.yace", FIX / f"{name}_ref.npz"
    pace_fixture(y), pace_fixture(r)
    return y, np.load(r)


def _atoms(ref, s):
    return Atoms(numbers=ref[f"Z_{s}"], positions=ref[f"pos_{s}"],
                 cell=ref[f"cell_{s}"], pbc=ref[f"pbc_{s}"])


@pytest.mark.parametrize("name", NAMES)
def test_parity_tight(name):
    y, ref = _fixture(name)
    calc = ACECalculator(str(y))
    for s in ref["struct_names"]:
        at = _atoms(ref, s)
        at.calc = calc
        n = len(at)
        assert abs(at.get_potential_energy() - ref[f"E_tight_{s}"]) / n < 1e-9, s
        np.testing.assert_allclose(at.get_forces(), ref[f"F_tight_{s}"], atol=1e-8, err_msg=s)
        if at.pbc.all():
            np.testing.assert_allclose(at.get_stress(), ref[f"S_tight_{s}"], atol=1e-9, err_msg=s)


@pytest.mark.parametrize("name", NAMES)
def test_parity_shipped_grid(name):
    """Gap to the C++ at the file's own deltaSplineBins: measured, then pinned."""
    y, ref = _fixture(name)
    calc = ACECalculator(str(y))
    worst = 0.0
    for s in ref["struct_names"]:
        at = _atoms(ref, s)
        at.calc = calc
        worst = max(worst, abs(at.get_potential_energy() - ref[f"E_ship_{s}"]) / len(at),
                    float(np.abs(at.get_forces() - ref[f"F_ship_{s}"]).max()))
    print(f"\n  {name}: worst |dE|/atom or |dF| vs shipped spline grid = {worst:.2e}")
    assert worst < SHIP_TOL


SHIP_TOL = 1e-5   # 10x the largest measured gap (9.8e-7, si_chebpow_fs: the sqrt
                  # embedding amplifies spline error; 2026-09-24, python-ace
                  # 0.2.7+192.g66c35ea, deltaSplineBins 0.001)


@pytest.mark.parametrize("name", ["gesi_sbessel", "sige_zbl"])
def test_forces_match_finite_differences(name):
    y, ref = _fixture(name)
    calc = ACECalculator(str(y))
    at = _atoms(ref, "close")[:12]
    at.calc = calc
    # directional derivative along random unit displacements: 2 energy calls per
    # direction instead of 6 per atom, and every force component contributes
    rng = np.random.default_rng(0)
    F, h = at.get_forces(), 1e-5
    for _ in range(3):
        v = rng.standard_normal(F.shape); v /= np.linalg.norm(v)
        e = []
        for sg in (1, -1):
            a = at.copy(); a.positions += sg * h * v; a.calc = calc
            e.append(a.get_potential_energy())
        assert abs(-(e[0] - e[1]) / (2 * h) - np.sum(F * v)) < 1e-6


def test_rotation_invariance():
    y, ref = _fixture("gesi_sbessel")
    at = _atoms(ref, "dimer_2.3") + _atoms(ref, "dimer_2.3").copy()
    at.positions[2:] += [0.3, 2.0, 0.7]
    at.calc = ACECalculator(str(y))
    E = at.get_potential_energy()
    b = at.copy(); b.rotate(37, [1, 2, 3], center="COM"); b.calc = at.calc
    assert abs(b.get_potential_energy() - E) < 1e-10


def test_mixed_rcut_neighbours_beyond_bond_cutoff():
    """gesi_sbessel: Ge-Ge rcut 4.2 < 5.0.  A Ge at 4.6 A from a Ge must not
    interact, although it is inside the neighbour-list cutoff."""
    y, _ = _fixture("gesi_sbessel")
    calc = ACECalculator(str(y))
    def e(d, zb):
        a = Atoms(numbers=[32, zb], positions=[[0, 0, 0], [d, 0, 0]], cell=np.eye(3) * 30, pbc=False)
        a.calc = calc
        return a.get_potential_energy(), a.get_forces()
    iso = Atoms(numbers=[32], positions=[[0, 0, 0]], cell=np.eye(3) * 30, pbc=False)
    iso.calc = calc
    E_iso = iso.get_potential_energy()
    E, F = e(4.6, 32)
    assert abs(E - 2 * E_iso) < 1e-12 and np.abs(F).max() < 1e-12
    E, _ = e(4.6, 14)                               # Ge-Si rcut 5.0: interacts
    assert abs(E - E_iso - _single(calc, 14)) > 1e-8


def _single(calc, z):
    a = Atoms(numbers=[z], positions=[[0, 0, 0]], cell=np.eye(3) * 30, pbc=False)
    a.calc = calc
    return a.get_potential_energy()


@pytest.mark.parametrize("name", ["si_chebpow_fs", "sige_zbl"])
def test_padded_and_neighbourless_nodes_finite(name):
    y, ref = _fixture(name)
    model, meta, _ = load(str(y))
    at = _atoms(ref, "bulk")
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    z2i = {z: i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers] + [0])      # one extra, isolated node
    pos = jnp.concatenate([jnp.asarray(at.positions), jnp.asarray([[50.0, 50, 50]])])
    send = jnp.concatenate([jnp.asarray(g.senders), jnp.asarray([len(at)] * 5)])
    recv = jnp.concatenate([jnp.asarray(g.receivers), jnp.asarray([len(at)] * 5)])
    shifts = jnp.concatenate([jnp.asarray(g.shifts), jnp.zeros((5, 3))])
    mask = jnp.concatenate([jnp.ones(len(g.senders), bool), jnp.zeros(5, bool)])
    f = lambda p: jnp.sum(model.energy_from_positions(p, nz, send, recv, mask, shifts))
    E, G = jax.value_and_grad(f)(pos)
    assert np.isfinite(E) and np.all(np.isfinite(G)) and np.all(G[-1] == 0)


def test_float32_close_to_float64():
    y, ref = _fixture("sige_distance")
    at = _atoms(ref, "bulk")
    at.calc = ACECalculator(str(y), dtype=jnp.float64)
    E64, F64 = at.get_potential_energy(), at.get_forces()
    at.calc = ACECalculator(str(y), dtype=jnp.float32)
    E32, F32 = at.get_potential_energy(), at.get_forces()
    assert np.isfinite(E32) and abs(E32 - E64) / abs(E64) < 1e-4
    assert np.all(np.isfinite(F32)) and np.abs(F32 - F64).max() < 1e-3


def test_dense_matches_sparse():
    y, ref = _fixture("gesi_sbessel")
    model, meta, _ = load(str(y))
    at = _atoms(ref, "bulk")
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    z2i = {z: i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers])
    s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    rij = jnp.asarray(g.rij)
    e_sp = model.site_energies(rij, nz[s], nz[r], s, len(at), nz)
    K = int(np.bincount(g.senders).max())
    n = len(at)
    R = np.full((n, K, 3), [meta["rcut"], 0, 0]); Zj = np.zeros((n, K), int); M = np.zeros((n, K), bool)
    fill = np.zeros(n, int)
    for e, (i, j) in enumerate(zip(g.senders, g.receivers)):
        R[i, fill[i]] = g.rij[e]; Zj[i, fill[i]] = z2i[int(at.numbers[j])]; M[i, fill[i]] = True; fill[i] += 1
    e_dn = model.site_energies_dense(jnp.asarray(R), jnp.broadcast_to(nz[:, None], (n, K)),
                                     jnp.asarray(Zj), jnp.asarray(M), nz)
    np.testing.assert_allclose(e_dn, e_sp, atol=1e-12)


def test_b_basis_methods_raise():
    y, _ = _fixture("si_chebexpcos")
    model, _, _ = load(str(y))
    with pytest.raises(NotImplementedError, match="B-basis"):
        model.site_descriptors(None, None, None, None, 0, None)


@pytest.mark.parametrize("name", ["si_chebpow_fs", "sige_zbl", "gesi_sbessel"])
def test_float32_graph_has_no_float64_arrays(name):
    """A float32 model must not promote any array to float64 (scalar literals are
    weakly typed and fine): f64 is 1/32-1/64 rate on consumer GPUs."""
    import re
    y, ref = _fixture(name)
    model, meta, _ = load(str(y), dtype=jnp.float32)
    at = _atoms(ref, "bulk")
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    z2i = {z: i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers])
    s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    jaxpr = str(jax.make_jaxpr(lambda x: model.energy_forces_virial(
        x, nz[s], nz[r], s, r, len(at), nz))(jnp.asarray(g.rij, jnp.float32)))
    assert re.findall(r"f64\[[0-9,]+\]", jaxpr) == []
