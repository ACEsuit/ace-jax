"""Descriptors match ACEpotentials.site_descriptors, and
ACECalculator works from a bare npz path.

Parity target is `ACEpotentials.site_descriptors` (src/descriptor.jl).  Its
layout is species-blocked: the centre species picks which (n_B, n_pair) blocks
are populated and the rest are zero (src/models/ace.jl:544-566).
"""
import pathlib

import jax
import numpy as np
import pytest

from conftest import species_index

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import ACECalculator, highest_precision, load, site_descriptors

TOL = 1e-10


@pytest.fixture
def case(npz):
    model, meta, z = load(npz)
    from ase import Atoms
    atoms = Atoms(numbers=np.asarray(z["test_Z"]),
                  positions=np.asarray(z["test_pos"]).T,
                  cell=np.asarray(z["test_cell"]).T,
                  pbc=np.asarray(z["test_pbc"]).astype(bool))
    return model, meta, z, atoms, np.asarray(z["test_desc"]).T


def test_descriptors_from_edge_list(case, npz):
    """Core path, against the Julia-supplied edge list."""
    model, meta, z, atoms, ref = case
    n = int(z["test_pos"].shape[1])
    send = jnp.asarray(z["test_edge_i"], jnp.int32)
    recv = jnp.asarray(z["test_edge_j"], jnp.int32)
    nz = species_index(z)
    with highest_precision():
        d = np.asarray(model.site_descriptors(
            jnp.asarray(z["test_edge_rij"].T), nz[send], nz[recv], send, n, nz))
    err = np.max(np.abs(d - ref))
    print(f"\n  {npz.stem}: shape {d.shape}  max|d| = {err:.3e} "
          f"(scale {np.max(np.abs(ref)):.4f}, rel {err/np.max(np.abs(ref)):.2e})")
    assert d.shape == ref.shape
    assert err < TOL


def test_descriptors_standalone(case, npz):
    """No ASE round-trip: positions/cell/species straight in."""
    model, meta, z, atoms, ref = case
    d = site_descriptors(npz, np.asarray(z["test_pos"]).T, np.asarray(z["test_Z"]),
                         np.asarray(z["test_cell"]).T, True)
    assert np.max(np.abs(d - ref)) < TOL


def test_descriptors_calculator_property(case, npz):
    model, meta, z, atoms, ref = case
    d = ACECalculator(npz).get_site_descriptors(atoms)
    assert np.max(np.abs(d - ref)) < TOL


def test_descriptor_layout_is_species_blocked(case, npz):
    """A single-species model populates exactly one block; the rest are zero.
    Guards the scatter: a wrong offset would still give plausible magnitudes."""
    model, meta, z, atoms, ref = case
    n_B, n_pair, nzc = meta["n_B"], meta["n_pair"], len(meta["elements"])
    assert ref.shape[1] == (n_B + n_pair) * nzc == meta["len_basis"]
    if nzc == 1:
        return
    d = np.asarray(ACECalculator(npz).get_site_descriptors(atoms))
    assert np.count_nonzero(d) <= d.shape[0] * (n_B + n_pair)


def test_descriptors_domain(case, npz):
    model, meta, z, atoms, ref = case
    dom = [0, 5, 9]
    d = site_descriptors(npz, np.asarray(z["test_pos"]).T, np.asarray(z["test_Z"]),
                         np.asarray(z["test_cell"]).T, True, domain=dom)
    assert d.shape == (len(dom), ref.shape[1])
    assert np.max(np.abs(d - ref[dom])) < TOL


def test_calculator_from_bare_path(case, npz):
    """The usability gate: a path and nothing else."""
    model, meta, z, atoms, ref = case
    atoms = atoms.copy()
    atoms.calc = ACECalculator(npz)
    dE = abs(atoms.get_potential_energy() - float(z["test_E"][0]))
    dF = np.max(np.abs(atoms.get_forces() - np.asarray(z["test_F"]).T))
    print(f"\n  {npz.stem}: |dE| = {dE:.3e}  |dF| = {dF:.3e}")
    assert dE < TOL and dF < TOL
    # cutoff and species came from the file, not from arguments
    assert atoms.calc.cutoff == float(meta["rcut"])
    assert atoms.calc._z2i == {int(e): i for i, e in enumerate(meta["elements"])}


def test_unknown_element_is_rejected(case, npz):
    """A silent wrong answer here would be worse than an exception."""
    model, meta, z, atoms, ref = case
    with pytest.raises(ValueError, match="not in model elements"):
        site_descriptors(npz, np.zeros((2, 3)), [79, 79])
