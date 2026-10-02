"""`aj eval` writes the input structures back as extxyz with the predictions added
(ace_energy, ace_forces, ace_stress; ace_*_std for UQ models), every original label
kept, and prints the per-config-type RMSE table."""
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR

from ace_jax.cli import main

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
KEYS = ["--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key", "dft_virial"]


def first_frames(path, n):
    """The first n frames of si_tiny_train.xyz, byte for byte (the isolated atom first)."""
    lines, out, i = XYZ.read_text().splitlines(keepends=True), [], 0
    for _ in range(n):
        k = int(lines[i]); out += lines[i:i + k + 2]; i += k + 2
    path.write_text("".join(out))
    return path


def _frames(path):
    import extxyz
    return list(extxyz.iread_dicts(str(path)))


def test_eval_writes_xyz_with_predictions_and_original_labels(tmp_path, capsys):
    from ase import Atoms
    from ace_jax import ACECalculator
    data = first_frames(tmp_path / "d.xyz", 6)
    assert main(["eval", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(data), *KEYS,
                 "--out", str(tmp_path / "pred.xyz")]) == 0
    from ace_jax.fit.xyz import read_extxyz
    src, out = _frames(data), _frames(tmp_path / "pred.xyz")
    assert len(out) == len(src) == 6
    calc = ACECalculator(str(FIXTURE_DIR / "si_fitted.npz"))
    for a, b, fr in zip(src, out, read_extxyz(data)):
        for k in a.info:                                        # every original label kept
            assert k in b.info
        assert b.info["config_type"] == a.info["config_type"]
        np.testing.assert_allclose(b.info["dft_energy"], a.info["dft_energy"], rtol=0, atol=1e-8)
        np.testing.assert_allclose(b.arrays["dft_force"], a.arrays["dft_force"], rtol=0, atol=1e-8)
        at = Atoms(numbers=fr.numbers, positions=fr.positions, cell=fr.cell, pbc=fr.pbc)
        at.calc = calc
        np.testing.assert_allclose(b.info["ace_energy"], at.get_potential_energy(), rtol=0, atol=1e-7)
        np.testing.assert_allclose(b.arrays["ace_forces"], at.get_forces(), rtol=0, atol=1e-7)
        if at.pbc.all():
            assert np.shape(b.info["ace_stress"]) == (3, 3)
    text = capsys.readouterr().out
    assert "RMSE" in text and "isolated_atom" in text and "config type" in text
    assert "wrote 6 configurations" in text


def test_eval_without_out_prints_the_table_only(tmp_path, capsys):
    data = first_frames(tmp_path / "d.xyz", 3)
    assert main(["eval", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(data), *KEYS]) == 0
    text = capsys.readouterr().out
    assert "config type" in text and "wrote" not in text


def test_eval_prefix(tmp_path):
    data = first_frames(tmp_path / "d.xyz", 2)
    assert main(["eval", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(data), *KEYS,
                 "--out", str(tmp_path / "p.xyz"), "--prefix", "mine_"]) == 0
    f = _frames(tmp_path / "p.xyz")[1]
    assert "mine_energy" in f.info and "mine_forces" in f.arrays and "ace_energy" not in f.info


def test_write_raw_round_trips_the_cell_and_every_label(tmp_path):
    """extxyz.iread_dicts returns the lattice column-major but write_dicts takes it
    row-major: written back unchanged, every non-symmetric cell came out transposed."""
    from ace_jax.fit.xyz import read_extxyz, read_raw, write_raw
    write_raw(tmp_path / "rt.xyz", read_raw(XYZ))
    for a, b in zip(read_extxyz(XYZ), read_extxyz(tmp_path / "rt.xyz")):
        np.testing.assert_array_equal(b.cell, a.cell)
        np.testing.assert_array_equal(b.pbc, a.pbc)
        np.testing.assert_allclose(b.positions, a.positions, rtol=0, atol=1e-8)
        assert set(b.info) == set(a.info) and set(b.arrays) == set(a.arrays)


def test_eval_output_keeps_the_input_cells(tmp_path):
    from ace_jax.fit.xyz import read_extxyz
    data = first_frames(tmp_path / "d.xyz", 4)
    assert main(["eval", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(data), *KEYS,
                 "--out", str(tmp_path / "p.xyz")]) == 0
    for a, b in zip(read_extxyz(data), read_extxyz(tmp_path / "p.xyz")):
        np.testing.assert_array_equal(b.cell, a.cell)
