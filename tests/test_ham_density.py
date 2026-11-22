"""Electron-density data, electron count, metrics, predictor and command line (`ham.density`).
Synthetic data only: random models, random overlaps, Gaussian radial tables."""
import jax

jax.config.update("jax_enable_x64", True)

import json
import pathlib
import warnings

import numpy as np
import pytest

from ace_jax.ham import density as D
from ace_jax.operators import centred as C
from ace_jax.operators import io
from ace_jax.operators.features import build_site_basis
from ace_jax.operators.orbitals import OrbitalBasis

CACHE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "ham" / "coupling_cache"
AUX = OrbitalBasis.of({14: (0, 0, 1, 2), 6: (0, 1)})
CHARGES = {14: np.array([1.3, 0.7] + [0.0] * 8), 6: np.array([2.1, 0.0, 0.0, 0.0])}


def structure(seed=0, n=8):
    rng = np.random.default_rng(seed)
    cell = 4.4 * np.eye(3) + 0.3 * rng.normal(size=(3, 3))
    return rng.uniform(size=(n, 3)) @ cell, cell, np.array([14, 6, 14, 6, 6, 14, 14, 6])[:n]


def spd(n, seed):
    A = np.random.default_rng(seed).normal(size=(n, n))
    S = A @ A.T / n + 0.5 * np.eye(n)
    d = np.sqrt(np.diag(S))
    return S / d[:, None] / d[None]


def record(seed=0, n=8, overlap=True, **info):
    """A record with random coefficients; its electron count is whatever they hold."""
    pos, cell, numbers = structure(seed, n)
    c = np.random.default_rng(seed + 1).normal(size=C.offsets(AUX, numbers)[-1])
    return D.DensityData(pos, cell, numbers, True, AUX, c, spd(len(c), seed) if overlap else None, CHARGES, D.electron_count(c, numbers, CHARGES), info)


def test_electron_count_and_conserve():
    """N is the direct sum over atoms of q . c; `conserve` returns the error before the
    correction and makes the count exact with a change along the charges only (two species)."""
    d = record(0)
    off = C.offsets(AUX, d.numbers)
    direct = sum(CHARGES[int(z)] @ d.coefficients[off[i]:off[i + 1]] for i, z in enumerate(d.numbers))
    assert np.isclose(D.electron_count(d.coefficients, d.numbers, CHARGES), direct, rtol=1e-14)
    target = direct + 0.37
    c, dN = D.conserve(d.coefficients, d.numbers, CHARGES, target)
    assert np.isclose(dN, -0.37, rtol=1e-12) and abs(D.electron_count(c, d.numbers, CHARGES) - target) < 1e-12
    q = np.concatenate([CHARGES[int(z)] for z in d.numbers])
    assert np.allclose(c - d.coefficients, 0.37 * q / (q @ q)) and not np.any((c - d.coefficients)[q == 0])
    other = np.linalg.lstsq(q[None], [0.37], rcond=None)[0]                 # the smallest Euclidean change that fixes the count
    assert np.allclose(c - d.coefficients, other)
    with pytest.raises(KeyError, match="32"):
        D.conserve(d.coefficients, np.where(d.numbers == 6, 32, d.numbers), CHARGES, 1.0)
    with pytest.raises(ValueError, match="charges of these atoms"):
        D.electron_count(d.coefficients[:-1], d.numbers, CHARGES)


def gaussians(seed=0):
    rng = np.random.default_rng(seed)
    r = np.linspace(0.0, 6.0, 30)
    return {z: (r, np.stack([r ** l * np.exp(-rng.uniform(0.3, 2.0) * r ** 2) for l in ls])) for z, ls in AUX.shells}


def test_file_round_trip(tmp_path):
    """The neutral file gives the record back: arrays bit for bit, the info, the radial functions;
    the overlap on request, or later."""
    info = dict(fit_metric="overlap", kpts=[2, 2, 2], kpt_offset=[0.25, 0.25, 0.25], smearing="gaussian", width=0.1, self_consistent=True)
    d = record(1, **info)._replace(shells=OrbitalBasis.of(dict(AUX.shells), gaussians()), pbc=(True, True, False))
    path = tmp_path / "a.npz"
    D.write(path, d)
    r = D.read(path)
    for k in ("positions", "cell", "numbers", "coefficients", "overlap"):
        assert np.array_equal(getattr(r, k), getattr(d, k)), k
    assert r.shells == AUX and r.pbc == (True, True, False) and r.info == info and r.nelectrons == d.nelectrons
    assert all(np.array_equal(r.charges[z], CHARGES[z]) for z in CHARGES)
    assert all(np.array_equal(r.shells.radials[z][i], d.shells.radials[z][i]) for z in (6, 14) for i in (0, 1))
    assert D.read(path, overlap=False).overlap is None
    lazy = D.read(path, overlap="lazy")
    assert callable(lazy.overlap) and np.array_equal(lazy.S(), d.overlap) and np.allclose(D.projections(lazy), d.overlap @ d.coefficients)
    meta = json.loads(bytes(np.load(path)["meta_json"]).decode())
    assert meta["schema_version"] == 1 and meta["kind"] == "density" and meta["m_order"] == D.M_ORDER and meta["shells"] == {"6": [0, 1], "14": [0, 0, 1, 2]}
    bare = D.DensityData(d.positions, d.cell, d.numbers, True, AUX, d.coefficients)
    D.write(tmp_path / "b.npz", bare)
    b = D.read(tmp_path / "b.npz")
    assert b.overlap is None and b.charges is None and b.nelectrons is None and b.info == {} and b.pbc is True and b.shells.radials is None
    with pytest.raises(ValueError, match="no overlap"):
        D.projections(b)
    with pytest.raises(ValueError, match="coefficients"):
        D.write(tmp_path / "c.npz", bare._replace(coefficients=d.coefficients[:-1]))
    np.savez(tmp_path / "x.npz", a=1)
    with pytest.raises(ValueError, match="not an ace-jax density file"):
        D.read(tmp_path / "x.npz")
    c = d.centred()
    assert c.basis == AUX and set(c.coefficients) == {"rho"} and c.pbc == (True, True, False)


def test_pair_record_and_deeph_directory(tmp_path):
    """A record that is not self-consistent keeps its flag and the link to the H built from it;
    a density in a structure's DeepH directory is found by `read` and `names`."""
    for k in range(2):
        h = tmp_path / "h" / f"s{k}"
        d = record(2 + k, overlap=False, self_consistent=False, hamiltonian=str(h), fit_metric="unknown")
        D.write(h / D.IN_DIRECTORY, d)
        r = D.read(h)
        assert r.info["self_consistent"] is False and r.info["hamiltonian"] == str(h) and np.array_equal(r.coefficients, d.coefficients)
    (tmp_path / "h" / "empty").mkdir()
    assert D.names(tmp_path / "h") == [str(tmp_path / "h" / "s0"), str(tmp_path / "h" / "s1")]
    D.write(tmp_path / "flat" / "b.npz", d)
    D.write(tmp_path / "flat" / "a.npz", d)
    assert [pathlib.Path(f).name for f in D.names(tmp_path / "flat")] == ["a.npz", "b.npz"]
    with pytest.raises(ValueError, match="no density files"):
        D.names(tmp_path / "h" / "empty")


def test_metrics():
    """The density error is sqrt(dc S dc); the per-l ratio is 1 for the mean coefficients; the
    summary has the median and the worst structure."""
    d = record(4)
    assert D.density_metrics(d, d.coefficients)["error"] == 0.0
    rng = np.random.default_rng(0)
    per = []
    for scale in (1e-3, 1e-2, 1e-1):
        dc = scale * rng.normal(size=d.coefficients.shape)
        m = D.density_metrics(d, d.coefficients + dc)
        assert np.isclose(m["error"], np.sqrt(dc @ d.overlap @ dc)) and np.isclose(m["relative"], m["error"] / np.sqrt(d.coefficients @ d.overlap @ d.coefficients))
        assert np.isclose(m["error_w"], np.linalg.norm(d.overlap @ dc)) and np.isclose(m["dN"], D.electron_count(dc, d.numbers, CHARGES))
        assert set(m["rmse_l"]) == set(m["rmse_w_l"]) == {0, 1, 2}
        per.append(m)
    s = D.summary(per)
    assert s["n"] == 3 and s["error_median"] == per[1]["error"] and s["error_worst"] == per[2]["error"] and s["dN_max"] == max(abs(m["dN"]) for m in per)
    assert len(D.format_summary(s)) == 5
    off, numbers = C.offsets(AUX, d.numbers), np.asarray(d.numbers)
    mean = np.zeros_like(d.coefficients)                     # the baseline: per species and l = 0 shell the mean, zero for l > 0
    for z in AUX.species:
        for s_, l in enumerate(AUX.ls(z)):
            if l == 0:
                rows = off[:-1][numbers == z] + AUX.shell_slice(z, s_).start
                mean[rows] = d.coefficients[rows].mean()
    m = D.density_metrics(d._replace(overlap=None), mean)
    assert all(np.isclose(v, 1.0) for v in m["rmse_l"].values()) and "error" not in m and "rmse_w_l" not in m


def test_fitted_charges_warn_and_conserve():
    """Without measured charges the fit to the counts reproduces them on the training structures,
    and says that the charges are not the functions' integrals."""
    data = [record(10 + k)._replace(charges=None) for k in range(6)]
    with pytest.warns(UserWarning, match="not the functions' integrals"):
        q = D.fit_charges(data)
    assert set(q) == {6, 14} and not np.any(q[14][2:]) and not np.any(q[6][1:])
    assert all(abs(D.electron_count(d.coefficients, d.numbers, q) - d.nelectrons) < 1e-8 for d in data)
    assert all(np.allclose(q[z], CHARGES[z], atol=1e-8) for z in q)          # six structures, three unknowns: here they are determined
    with pytest.raises(ValueError, match="nelectrons"):
        D.fit_charges([data[0]._replace(nelectrons=None)])
    assert D.electrons_per_atom([d._replace(nelectrons=float(np.sum(d.numbers))) for d in data]) == {6: 6.0, 14: 14.0}
    mixed = [record(20 + k, n)._replace(nelectrons=4.0 * np.sum(structure(0, n)[2] == 14) + 4.0 * np.sum(structure(0, n)[2] == 6)) for k, n in enumerate((8, 5, 3))]
    e = D.electrons_per_atom(mixed)
    assert np.isclose(e[14], 4.0) and np.isclose(e[6], 4.0)


def cli_basis():
    """The site basis `aj density fit --site-rcut 4 --site-radials 4` builds for AUX."""
    return build_site_basis([6, 14], order=1, totaldegree=40, lmax_feat=2, rcut=4.0, wL=0.5, nmax_by_order=(4,), lmax_by_order=(2,), natural_parity=True).with_table()


@pytest.fixture(scope="module")
def synthetic():
    """A random model on the command's basis, and records with its coefficients, a random
    overlap, charges and the electron count the coefficients hold."""
    sb = cli_basis()
    rng = np.random.default_rng(0)
    coef = {(z, s): rng.normal(size=(1, C.n_columns(sb, l))) for z in AUX.species for s, l in enumerate(AUX.ls(z))}
    model = C.CentredModel(basis=sb, coef=coef, shells=AUX, channels=("rho",))
    out = []
    for k in range(5):
        pos, cell, numbers = structure(30 + k)
        c = model.predict(pos, cell, numbers)["rho"]
        out.append(D.DensityData(pos, cell, numbers, True, AUX, c, spd(len(c), k), CHARGES, D.electron_count(c, numbers, CHARGES), dict(fit_metric="overlap", kpts=[2, 2, 2])))
    return model, out


def test_predictor_from_a_model_file(synthetic, tmp_path):
    """The predictor of a file: the model's coefficients corrected to the count, dN in the info;
    a file without charges predicts uncorrected."""
    model, data = synthetic
    rec = dict(charges={str(z): list(q) for z, q in CHARGES.items()}, electrons={"14": 3.1, "6": 2.2})
    io.save(tmp_path / "m.npz", centred=model, info=dict(density=rec))
    p = D.DensityPredictor(tmp_path / "m.npz")
    from ase import Atoms
    d = data[0]
    at = Atoms(numbers=d.numbers, positions=d.positions, cell=d.cell, pbc=True)
    out = D.predict(tmp_path / "m.npz", at)
    n = 3.1 * 4 + 2.2 * 4
    assert out.nelectrons == pytest.approx(n) and abs(D.electron_count(out.coefficients, d.numbers, CHARGES) - n) < 1e-10
    assert out.info["dN"] == pytest.approx(d.nelectrons - n, abs=1e-8) and out.info["corrected"] and out.info["self_consistent"] is False
    same = p.predict(at, d.nelectrons)
    assert abs(same.info["dN"]) < 1e-8 and np.allclose(same.coefficients, d.coefficients, atol=1e-8) and same.shells == AUX and same.overlap is None
    io.save(tmp_path / "bare.npz", centred=model)
    bare = D.DensityPredictor(tmp_path / "bare.npz").predict(at)
    assert bare.info["dN"] is None and not bare.info["corrected"] and np.allclose(bare.coefficients, d.coefficients, atol=1e-8)
    assert bare.nelectrons == 4 * 14 + 4 * 6                 # no record: the atomic numbers


def test_command_line(synthetic, tmp_path, capsys):
    """`aj density fit`, `predict` and `aj eval` on four small synthetic structures (and a fifth
    as the validation set): the fit on the generating model's basis finds its coefficients."""
    from ace_jax.cli import main
    model, data = synthetic
    for k, d in enumerate(data):
        D.write(tmp_path / ("train" if k < 4 else "val") / f"s{k}.npz", d)
    out = str(tmp_path / "model.npz")
    basis = ["--site-rcut", "4", "--site-radials", "4"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # one fit_metric, one k-point grid, charges in the data: nothing to warn of
        warnings.filterwarnings("ignore", message=".*do not fix the electrons per atom.*")     # but one composition, and counts that are not integers
        main(["density", "fit", "--data", str(tmp_path / "train"), "--out", out, "--validation", str(tmp_path / "val"), *basis])
    text = capsys.readouterr().out
    assert "4 structures, 32 atoms" in text and "metric overlap" in text and "224 rows, 51 parameters" in text and "validation set" in text
    f = io.load(out)
    info = f.meta["info"]
    assert f.onsite is None and f.centred.shells == AUX and info["metric"] == "overlap" and info["n_coef"] == 51 and info["scaled"] == 1e-6
    assert info["density"]["electrons"].keys() == {"6", "14"} and np.allclose(info["density"]["charges"]["14"], CHARGES[14])
    assert info["validation"]["metrics"]["relative_worst"] < 1e-3 and info["fit_metric"] == "overlap"
    pos, cell, numbers = structure(77)
    a, b = model.predict(pos, cell, numbers)["rho"], f.centred.predict(pos, cell, numbers)["rho"]
    assert np.abs(a - b).max() < 1e-3 * np.abs(a).max()
    main(["density", "predict", "--model", out, "--data", str(tmp_path / "val"), "--out", str(tmp_path / "pred")])
    text = capsys.readouterr().out
    assert "s4: 8 atoms, electron count error before the correction" in text
    pred = D.read(tmp_path / "pred" / "s4.npz")
    assert abs(D.electron_count(pred.coefficients, pred.numbers, CHARGES) - data[4].nelectrons) < 1e-9 and abs(pred.info["dN"]) < 1e-2
    assert np.abs(pred.coefficients - data[4].coefficients).max() < 1e-3 * np.abs(data[4].coefficients).max()
    main(["eval", "--model", out, "--data", str(tmp_path / "val"), "--metrics-out", str(tmp_path / "metrics.json")])
    text = capsys.readouterr().out
    assert "1 reference structures" in text and "density error sqrt(dc S dc)" in text and "electron count error before the correction" in text
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["summary"]["relative_worst"] == pytest.approx(info["validation"]["metrics"]["relative_worst"]) and m["structures"][0]["name"] == "s4"
    plain = str(tmp_path / "plain.npz")
    main(["density", "fit", "--data", str(tmp_path / "train"), "--out", plain, "--metric", "plain", "--prior", "1e3", "--max-structures", "3", *basis])
    info = io.load(plain).meta["info"]
    assert info["metric"] == "plain" and info["n_train"] == 3 and info["scaled"] is None and info["prior"] == "1e3"
    capsys.readouterr()
    for d in data[:2]:                                       # no overlap, a second k-point grid, no charges
        D.write(tmp_path / "bare" / f"{id(d)}.npz", d._replace(overlap=None, charges=None, info=dict(d.info, kpts=[3, 3, 3] if d is data[0] else [2, 2, 2])))
    with pytest.warns(UserWarning) as rec:
        main(["density", "fit", "--data", str(tmp_path / "bare"), "--out", plain, *basis])
    msgs = " ".join(str(w.message) for w in rec)
    assert "different k-points or smearing" in msgs and "not the functions' integrals" in msgs and "metric plain" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["density", "fit", "--data", str(tmp_path / "bare"), "--out", plain, "--metric", "overlap", *basis])
    assert "--metric overlap" in capsys.readouterr().err
