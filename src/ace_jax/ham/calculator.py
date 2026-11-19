"""ASE calculator for a Hamiltonian model: band energy, forces and stress, and the electronic
structure they come from (eigenvalues, Fermi level, the H and S blocks).

    from ace_jax.ham.calculator import HamiltonianCalculator
    calc = HamiltonianCalculator("model.npz", kpts=(4, 4, 4), nelectrons={79: 11}, width=0.1)     # or (onsite, offsite)
    atoms.calc = calc
    atoms.get_forces(); atoms.get_stress()
    calc.get_eigenvalues(kpt=0); calc.get_fermi_level()
    H = calc.get_hamiltonian(atoms)          # {(Rx, Ry, Rz, i, j): (n_i, n_j) array}
    calc.write_results("pred")               # DeepH layout, as MACE-H's calculator writes

The layout follows MACE-H's `MACEHCalculator` (results["hamiltonian"], `get_hamiltonian`,
`write_results`), with what a Hamiltonian that is differentiable in the positions adds.  The
energy is the BAND energy of the predicted H and S, sum_k w_k sum_n f e, and the forces and stress
are the derivatives of its free energy (`HamiltonianPredictor.band_energy`: the trace form, the
eigensolver is not differentiated).  It is not a total energy: there is no repulsive or
double-counting term, so it is a building block (e.g. beside a pair potential), not a potential."""
import os

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from ..operators.maps import make_room
from .operator import SMEARINGS, HamiltonianPredictor


class HamiltonianCalculator(Calculator):
    """predictor: a model file (`operators.io.save`; `threads=` for its off-site evaluation), a
    `HamiltonianPredictor`, or (onsite model, offsite model).
    kpts: (n1, n2, n3), a Gamma-centred grid of fractional k-points i / n, or an (n_k, 3) array of
    fractional k-points; weights: their weights (default equal).
    nelectrons: the electrons the bands hold: None, the atomic numbers (an all-electron basis); a
    number, per cell; or {Z: electrons per atom} (e.g. {79: 11} for a valence-only gold basis).
    smearing, width: "fermi-dirac" (width = k_B T) or "gaussian", in the energy unit of the model.
    overlap: the overlap channel ("S"), or None for an orthonormal basis; channel: the Hamiltonian
    channel ("H"; "H_up" or "H_dn" of a collinear-spin model, one spin per calculator, with
    degeneracy 1).  device: build the neighbour lists on the GPU.
    core_bands: {Z: bands per atom} (or a number per cell) left out of the band energy, forces and
    stress: with an all-electron basis the sum over core levels moves by an eV per atom between
    structures and a residual potential cannot follow it, where the valence band energy leaves a
    learnable remainder (docs/dev/ham-results.md).  The eigenvalues and the Fermi level keep them.

    Properties: energy (band energy), free_energy (energy - T S; what forces and stress
    differentiate), forces, stress (ASE's sign and Voigt order), eigenvalues (n_k, n_orb),
    fermi_level; hamiltonian and overlap (dictionaries of blocks) only when asked for."""

    implemented_properties = ["energy", "free_energy", "forces", "stress", "eigenvalues", "fermi_level", "hamiltonian", "overlap"]

    def __init__(self, predictor, kpts=(1, 1, 1), weights=None, nelectrons=None, width=0.1, smearing="fermi-dirac",
                 overlap="S", channel="H", degeneracy=2.0, device=False, core_bands=None, threads=1, **kwargs):
        super().__init__(**kwargs)
        if isinstance(predictor, (str, os.PathLike)):
            predictor = HamiltonianPredictor.from_file(predictor, threads=threads)
        self.predictor = predictor if isinstance(predictor, HamiltonianPredictor) else HamiltonianPredictor(*predictor, threads=threads)
        if smearing not in SMEARINGS:
            raise ValueError(f"smearing must be one of {SMEARINGS}, got {smearing!r}")
        k = np.asarray(kpts, float)
        if k.shape == (3,) and np.all(k == np.rint(k)) and np.all(k >= 1):
            n1, n2, n3 = (int(v) for v in k)
            k = np.array([[a / n1, b / n2, c / n3] for a in range(n1) for b in range(n2) for c in range(n3)])
        self.kpts = np.atleast_2d(k)
        self.weights = np.full(len(self.kpts), 1.0 / len(self.kpts)) if weights is None else np.asarray(weights, float)
        self.nelectrons, self.width, self.smearing = nelectrons, float(width), smearing
        self.overlap, self.channel, self.degeneracy, self.device = overlap, channel, float(degeneracy), bool(device)
        self.core_bands = core_bands

    def electrons(self, numbers):
        """The number of electrons of a cell with these atomic numbers."""
        if self.nelectrons is None:
            return float(np.sum(numbers))
        if isinstance(self.nelectrons, dict):
            return float(sum(self.nelectrons[int(z)] for z in numbers))
        return float(self.nelectrons)

    def core(self, numbers):
        """The number of lowest bands of the cell left out of the band energy."""
        if self.core_bands is None:
            return 0
        if isinstance(self.core_bands, dict):
            return int(sum(self.core_bands.get(int(z), 0) for z in numbers))
        return int(self.core_bands)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms
        if abs(a.cell.volume) < 1e-12:
            raise ValueError("HamiltonianCalculator needs a cell (put an isolated system in a box)")
        make_room(58_000)                                # the band-energy program of a four-element model: ~50,000 regions
        out = self.predictor.band_energy(a.positions, np.asarray(a.cell), a.numbers, self.kpts, self.electrons(a.numbers),
                                         width=self.width, smearing=self.smearing, weights=self.weights, overlap=self.overlap,
                                         channel=self.channel, degeneracy=self.degeneracy, device=self.device, pbc=a.pbc,
                                         core_bands=self.core(a.numbers))
        self.results = {"energy": float(out.energy), "free_energy": float(out.free_energy),
                        "forces": -np.asarray(out.gradient, float),
                        "stress": full_3x3_to_voigt_6_stress(np.asarray(out.stress, float)),
                        "eigenvalues": np.asarray(out.eigenvalues, float), "fermi_level": float(out.fermi_level)}
        if "hamiltonian" in properties or "overlap" in properties:
            blocks = self._blocks(a)
            self.results["hamiltonian"] = blocks[self.channel]
            if self.overlap:
                self.results["overlap"] = blocks[self.overlap]

    def _blocks(self, atoms):
        p = self.predictor
        make_room(45_000)                                # a program per species pair (`operators.maps`)
        on = p.onsite(atoms.positions, np.asarray(atoms.cell), atoms.numbers, atoms.pbc)
        off = p.offsite(atoms.positions, np.asarray(atoms.cell), atoms.numbers, atoms.pbc)
        return {c: {**off[c], **on[c]} for c in p.channels}

    # -- electronic structure, in the names ASE's band-structure and DOS tools call
    def get_hamiltonian(self, atoms=None):
        """{(Rx, Ry, Rz, i, j): (n_i, n_j)} blocks of H: atom i in the home cell, atom j in cell R,
        0-based, orbitals in ace-jax's order (m = -l..l within a shell)."""
        return self.get_property("hamiltonian", atoms)

    def get_overlap(self, atoms=None):
        return self.get_property("overlap", atoms)

    def get_eigenvalues(self, kpt=0, spin=0):
        return self.get_property("eigenvalues")[kpt]

    def get_fermi_level(self):
        return self.get_property("fermi_level")

    def get_ibz_k_points(self):
        return self.kpts.copy()

    get_bz_k_points = get_ibz_k_points

    def get_k_point_weights(self):
        return self.weights.copy()

    def get_number_of_spins(self):
        return 1

    def get_number_of_bands(self):
        return self.get_property("eigenvalues").shape[1]

    def write_results(self, output_dir, atoms=None, overlap=True):
        """Write the predicted blocks in the DeepH layout (`ham.deeph.write`): the structure files,
        `hamiltonians_pred.h5` and, with a model that has an overlap channel, `overlaps_pred.h5`."""
        from ..operators.data import OperatorData
        from . import deeph
        atoms = self.atoms if atoms is None else atoms
        blocks = self._blocks(atoms)
        files = {self.channel: "hamiltonians_pred.h5"}
        if overlap and self.overlap:
            files[self.overlap] = "overlaps_pred.h5"
        data = OperatorData(np.asarray(atoms.cell), atoms.positions, atoms.numbers, self.predictor.orbitals,
                            {c: blocks[c] for c in files})
        deeph.write(output_dir, data, files=files)
