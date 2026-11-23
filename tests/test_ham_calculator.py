"""`HamiltonianCalculator`: band energy, forces and stress through ASE, with eigenvalues, the Fermi
level and the H blocks, on a random two-element model."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from ase import Atoms
from conftest import require_optional
from test_ham_operator import _env_predictor, models, structure  # noqa: F401 (models: a fixture)

from ace_jax.ham import deeph
from ace_jax.ham.calculator import HamiltonianCalculator


@pytest.fixture(scope="module")
def setup(models):  # noqa: F811
    pos, cell, numbers = structure()
    calc = HamiltonianCalculator(_env_predictor(models), kpts=(2, 1, 2), nelectrons=40, width=0.2, overlap=None)
    return Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True), calc


def test_energy_forces_and_stress(setup):
    """Forces and stress are the derivatives of the free energy (central differences through new
    calculations), the occupations hold the electrons, and the Fermi level is among the bands."""
    atoms, calc = setup
    a = atoms.copy(); a.calc = calc
    F, s, e = a.get_forces(), a.get_stress(voigt=False), a.get_potential_energy()
    fe = a.get_potential_energy(force_consistent=True)
    assert F.shape == (4, 3) and np.abs(F.sum(axis=0)).max() < 1e-8 * np.abs(F).max() and fe < e
    ev, mu = calc.results["eigenvalues"], calc.get_fermi_level()
    assert ev.shape == (4, calc.get_number_of_bands()) and ev.min() < mu < ev.max()
    occ = 2.0 / (1.0 + np.exp((ev - mu) / 0.2))
    assert abs((occ * calc.get_k_point_weights()[:, None]).sum() - 40) < 1e-8
    assert abs((occ * ev * calc.get_k_point_weights()[:, None]).sum() - e) < 1e-8
    assert np.allclose(calc.get_eigenvalues(kpt=2), ev[2]) and calc.get_ibz_k_points().shape == (4, 3)

    def free(x, c):
        b = Atoms(numbers=atoms.numbers, positions=x, cell=c, pbc=True); b.calc = calc
        return b.get_potential_energy(force_consistent=True)
    h = 1e-4
    for i, d in ((0, 0), (2, 1), (3, 2)):
        dx = np.zeros_like(atoms.positions); dx[i, d] = h
        fd = -(free(atoms.positions + dx, atoms.cell[:]) - free(atoms.positions - dx, atoms.cell[:])) / (2 * h)
        assert abs(fd - F[i, d]) < 1e-5 * max(1.0, np.abs(F).max()), (i, d, fd, F[i, d])
    V = atoms.get_volume()
    for (p, q) in ((0, 0), (1, 2)):
        eps = np.zeros((3, 3)); eps[p, q] += 0.5; eps[q, p] += 0.5
        up, dn = np.eye(3) + h * eps, np.eye(3) - h * eps
        fd = (free(atoms.positions @ up.T, atoms.cell[:] @ up.T) - free(atoms.positions @ dn.T, atoms.cell[:] @ dn.T)) / (2 * h) / V
        assert abs(fd - s[p, q]) < 1e-5 * max(1.0, np.abs(s).max()), (p, q, fd, s[p, q])


def test_hamiltonian_blocks_and_files(setup, tmp_path):
    """`get_hamiltonian`: every bond's reverse is its transpose, on-site blocks are symmetric; and
    `write_results` writes a DeepH directory that reads back to the same blocks."""
    require_optional("h5py")
    atoms, calc = setup
    full = HamiltonianCalculator(calc.predictor, kpts=(1, 1, 1), nelectrons=40, overlap=None)
    full.overlap = "S"                                   # the blocks of both channels (S is not solved with here)
    a = atoms.copy(); a.calc = full
    H = full.get_hamiltonian(a)
    assert all(np.allclose(H[(-k[0], -k[1], -k[2], k[4], k[3])], v.T, atol=1e-10) for k, v in H.items())
    assert (0, 0, 0, 0, 0) in H and len(H) > 20
    full.write_results(tmp_path / "pred", a)
    back = deeph.read(tmp_path / "pred", operators=(("H", "hamiltonians_pred.h5"), ("S", "overlaps_pred.h5")))
    assert set(back.blocks["H"]) == set(H) and all(np.allclose(back.blocks["H"][k], v, atol=1e-12) for k, v in H.items())
    assert np.allclose(back.positions, a.positions) and list(back.numbers) == list(a.numbers)


def test_electron_counts():
    c = HamiltonianCalculator.__new__(HamiltonianCalculator)
    for spec, want in ((None, 14 + 6 + 14), (7.0, 7.0), ({14: 4, 6: 4}, 12.0)):
        c.nelectrons = spec
        assert c.electrons(np.array([14, 6, 14])) == want
    from types import SimpleNamespace
    c.predictor, c.degeneracy, c.nelectrons = SimpleNamespace(orbitals=SimpleNamespace(n_orb=lambda z: 4)), 2.0, None
    with pytest.raises(ValueError, match="do not fit"):     # every electron of the atoms, a valence-only basis: 34 into 24 states
        c.electrons(np.array([14, 6, 14]))
    c.nelectrons = {14: 4, 6: 4}
    assert c.electrons(np.array([14, 6, 14])) == 12.0


def test_core_bands_are_left_out_of_the_energy_and_its_derivatives(setup):
    """`core_bands`: the band energy is the sum over the remaining bands with the same Fermi level,
    and the forces are still the derivative of the free energy.  That holds when the bands left
    out are fully occupied (their occupations do not respond), so the smearing here is narrow."""
    atoms, calc = setup
    kw = dict(kpts=(2, 1, 2), nelectrons=40, width=0.01, overlap=None)
    full = HamiltonianCalculator(calc.predictor, **kw)
    val = HamiltonianCalculator(calc.predictor, core_bands={14: 2, 6: 1}, **kw)
    assert val.core(atoms.numbers) == 6 and full.core(atoms.numbers) == 0
    a, b = atoms.copy(), atoms.copy()
    a.calc, b.calc = full, val
    a.get_potential_energy()
    ev, mu, w = full.results["eigenvalues"], full.results["fermi_level"], full.get_k_point_weights()
    occ = 2.0 / (1.0 + np.exp(np.clip((ev - mu) / 0.01, -500, 500)))
    assert np.allclose(occ[:, :6], 2.0)                                   # the bands left out are full
    assert abs(b.get_potential_energy() - (occ * ev * w[:, None])[:, 6:].sum()) < 1e-8 and abs(val.get_fermi_level() - mu) < 1e-10
    F = b.get_forces()
    h = 1e-4

    def free(x):
        c = Atoms(numbers=atoms.numbers, positions=x, cell=atoms.cell, pbc=True); c.calc = val
        return c.get_potential_energy(force_consistent=True)
    for i, d in ((1, 2), (2, 0)):
        dx = np.zeros_like(atoms.positions); dx[i, d] = h
        fd = -(free(atoms.positions + dx) - free(atoms.positions - dx)) / (2 * h)
        assert abs(fd - F[i, d]) < 1e-5 * max(1.0, np.abs(F).max()), (i, d, fd, F[i, d])


def test_calculator_from_a_model_file_and_aj_eval(models, tmp_path, capsys):  # noqa: F811
    """A model file (`operators.io`) as the calculator's predictor gives the models' results, and
    `aj eval` on it: against reference blocks (here the model's own, H shifted by a constant
    times S: zero block and band errors but for the shift, which moves only the Fermi level) and
    on an extxyz file (band energy, Fermi level, forces and stress written back)."""
    require_optional("h5py")                     # --blocks-out and aj ham predict write the DeepH layout
    from ase_extxyz.io import read_cextxyz, write_cextxyz

    from ace_jax.cli import main
    from ace_jax.ham import cache
    from ace_jax.operators import io
    from ace_jax.operators.data import OperatorData
    path = str(tmp_path / "ham.npz")
    io.save(path, *models)
    kw = dict(kpts=(2, 1, 2), nelectrons=40, width=0.2)
    ref, calc = HamiltonianCalculator(models, **kw), HamiltonianCalculator(path, **kw)
    frames, shift = [], 0.25
    (tmp_path / "ref").mkdir()
    for i in range(2):
        pos, cell, numbers = structure(i)
        at = Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True)
        a, b = at.copy(), at.copy()
        a.calc, b.calc = ref, calc
        assert abs(a.get_potential_energy() - b.get_potential_energy()) < 1e-11 and np.allclose(a.get_forces(), b.get_forces(), atol=1e-10)   # bit-equal on a CPU; a GPU's eigensolver is not
        H, S = ref.get_hamiltonian(at), ref.get_overlap(at)
        cache.save(str(tmp_path / "ref" / f"{i}.npz"), OperatorData(cell, pos, numbers, ref.predictor.orbitals,
                                                                   {"H": {k: v + shift * S[k] for k, v in H.items()}, "S": S}))
        frames.append((at, a.get_potential_energy(), ref.get_fermi_level(), a.get_forces()))
    args = ["--kpts", "2", "1", "2", "--nelectrons", "40", "--width", "0.2"]
    main(["eval", "--model", path, "--data", str(tmp_path / "ref"), *args, "--window", "-50", "50", "--metrics-out", str(tmp_path / "m.json"), "--blocks-out", str(tmp_path / "pred")])
    out = capsys.readouterr().out
    row = {tuple(ln.split()[:2]): [float(v) for v in ln.split()[2:]] for ln in out.splitlines() if ln.startswith("  ")}
    assert max(row[("S", "on-site")] + row[("S", "off-site")]) < 1e-9 and row[("H", "off-site")][0] > 0
    import json
    m = json.load(open(tmp_path / "m.json"))["summary"]            # the shift: in the common-zero error and the Fermi level only
    assert m["aligned_rmse"] < 1e-8 and abs(m["band_rmse"] - shift) < 1e-8 and abs(m["fermi_rmse"] - shift) < 1e-8 and m["n"] == 2
    assert "bands at a common zero" in out and not m["outliers"]
    assert (tmp_path / "pred" / "0001" / "hamiltonians_pred.h5").exists()                       # the layout: the write_results test
    write_cextxyz(str(tmp_path / "s.xyz"), [f[0] for f in frames])
    main(["eval", "--model", path, "--data", str(tmp_path / "s.xyz"), *args, "--out", str(tmp_path / "o.xyz")])
    for f, (_, e, mu, F) in zip(read_cextxyz(str(tmp_path / "o.xyz"), index=":", create_calc=False), frames):
        assert abs(f.info["ace_band_energy"] - e) < 1e-6 and abs(f.info["ace_fermi_level"] - mu) < 1e-6      # the file's digits
        assert np.allclose(f.arrays["ace_forces"], F, atol=1e-6) and f.info["ace_stress"].shape == (3, 3)


def test_aj_ham_fit_predict_bands(models, tmp_path, capsys):  # noqa: F811
    """`aj ham fit` writes a model file from structures with blocks; fitted again to its own
    predictions with the same settings it reproduces them (the validation table and the file's
    record).  `aj ham predict` writes its blocks in the DeepH layout, and `aj ham bands` the bands
    on a path and a grid, the reference's from the structure's blocks beside them."""
    require_optional("h5py")                     # --blocks-out and aj ham predict write the DeepH layout
    from ase_extxyz.io import write_cextxyz

    from ace_jax.cli import main
    from ace_jax.ham import cache
    from ace_jax.operators import io
    from ace_jax.operators.data import OperatorData
    gen = HamiltonianCalculator(models, kpts=(1, 1, 1), nelectrons=40)
    orb, frames = gen.predictor.orbitals, []
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    for i in range(3):
        pos, cell, numbers = structure(i)
        at = Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True)
        frames.append(at)
        cache.save(str(tmp_path / "a" / f"{i}.npz"), OperatorData(cell, pos, numbers, orb, {"H": gen.get_hamiltonian(at), "S": gen.get_overlap(at)}))
    basis = ["--site-rcut", "4.0", "--site-radials", "3", "--bond-radials", "3", "--coupled", "1", "1", "--env", "4", "2", "1", "1", "1"]
    spec = ["--kpts", "2", "1", "2", "--nelectrons", "40", "--width", "0.2", "--window", "-50", "50"]
    first = str(tmp_path / "first.npz")
    main(["ham", "fit", "--data", str(tmp_path / "a"), "--out", first, *basis])
    assert "wrote" in capsys.readouterr().out and io.load(first).meta["info"]["n_train"] == 3
    own = HamiltonianCalculator(first, kpts=(1, 1, 1), nelectrons=40)
    for i, at in enumerate(frames):
        cache.save(str(tmp_path / "b" / f"{i}.npz"), OperatorData(np.asarray(at.cell), at.positions, at.numbers, orb,
                                                                 {"H": own.get_hamiltonian(at), "S": own.get_overlap(at)}))
    second = str(tmp_path / "second.npz")
    main(["ham", "fit", "--data", str(tmp_path / "b"), "--out", second, "--validation", str(tmp_path / "b"), *basis, *spec,
          "--overlap-threshold", "1e-8"])                           # below the spectrum of S: nothing is left out, and the option reaches the score
    out = capsys.readouterr().out
    info = io.load(second).meta["info"]
    assert "below 1e-08 left out" in out and info["validation"]["metrics"]["overlap_threshold"] == 1e-8
    assert "bands at a common zero" in out and max(v["max"] for v in info["validation"]["blocks"].values()) < 1e-4       # the ridge on designs with more columns than rows
    assert info["validation"]["metrics"]["band_rmse"] < 1e-3 and info["prior"] == "none"
    with pytest.raises(SystemExit):                                # a channel the data does not have
        main(["ham", "fit", "--data", str(tmp_path / "b"), "--out", second, "--channels", "H_up"])
    honly = str(tmp_path / "honly.npz")                           # a model of H alone, scored on data with S: both spectra with the reference S
    main(["ham", "fit", "--data", str(tmp_path / "b"), "--out", honly, "--channels", "H", *basis])
    main(["eval", "--model", honly, "--data", str(tmp_path / "b"), *spec, "--metrics-out", str(tmp_path / "h.json"),
          "--overlap-threshold", "1e-8"])
    import json
    mh = json.load(open(tmp_path / "h.json"))
    assert mh["summary"]["overlap_threshold"] == 1e-8
    assert "with the reference S" in capsys.readouterr().out and mh["summary"]["overlap"] == "reference" and mh["summary"]["band_rmse"] < 1e-3
    capsys.readouterr()
    write_cextxyz(str(tmp_path / "s.xyz"), frames)
    main(["ham", "predict", "--model", second, "--data", str(tmp_path / "s.xyz"), "--out", str(tmp_path / "pred")])
    got = deeph.read(str(tmp_path / "pred" / "0002"), operators=(("H", "hamiltonians_pred.h5"), ("S", "overlaps_pred.h5")))
    ref = cache.load(str(tmp_path / "b" / "2.npz"))[0]
    assert all(np.allclose(got.blocks[c][k], v, atol=1e-4) for c in ("H", "S") for k, v in ref.blocks[c].items())
    main(["ham", "bands", "--model", second, "--structure", str(tmp_path / "b"), "--index", "1", "--npoints", "12", "--out", str(tmp_path / "bands.npz"),
          *[x for x in spec if x not in ("--nelectrons", "40")]])        # the electron count: the one the fit recorded
    b = np.load(tmp_path / "bands.npz")
    assert info["nelectrons"] == "40" and b["nelectrons"] == 40
    assert b["bands"].shape == b["bands_ref"].shape and len(b["bands"]) == 12 and b["grid_bands"].shape[0] == 4
    assert np.allclose(b["bands"], b["bands_ref"], atol=1e-3) and abs(b["fermi_level"] - b["fermi_level_ref"]) < 1e-3 and b["metric_band_rmse"] < 1e-3
    pytest.importorskip("matplotlib")
    main(["ham", "bands", "--model", second, "--structure", str(tmp_path / "s.xyz"), "--npoints", "12", "--plot", str(tmp_path / "bands.png"), *spec])
    assert (tmp_path / "bands.png").stat().st_size > 1000
