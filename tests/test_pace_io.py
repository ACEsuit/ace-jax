import pathlib
import numpy as np
import pytest
from conftest import pace_fixture
from ace_jax.eval.pace_io import load_tree, parse_yace

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"

MINI = """\
elements: [Si]
E0: [-1.5]
deltaSplineBins: 0.001
embeddings:
  0: {ndensity: 1, FS_parameters: [1, 1], npoti: FinnisSinclairShiftedScaled, rho_core_cutoff: 100000, drho_core_cutoff: 250}
bonds:
  [0, 0]: {radbasename: ChebExpCos, radparameters: [5.25], radcoefficients: [[[1, 0.5], [0.3, 0.2]]], prehc: 0, lambdahc: 1, rcut: 5, dcut: 0.01, rcut_in: 0, dcut_in: 0, inner_cutoff_type: density, nradmax: 1, lmax: 1, nradbasemax: 2}
functions:
  0:
    - {mu0: 0, rank: 1, ndensity: 1, num_ms_combs: 1, mus: [0], ns: [1], ls: [0], ms_combs: [0], ctildes: [0.7]}
    - {mu0: 0, rank: 2, ndensity: 1, num_ms_combs: 3, mus: [0, 0], ns: [1, 1], ls: [1, 1], ms_combs: [-1, 1, 0, 0, 1, -1], ctildes: [0.1, -0.2, 0.1]}
"""


@pytest.fixture
def mini(tmp_path):
    p = tmp_path / "mini.yace"
    p.write_text(MINI)
    return p


def test_tuple_bond_keys(mini):
    t = load_tree(mini)
    assert (0, 0) in t["bonds"] and t["embeddings"][0]["ndensity"] == 1


def test_parse_mini(mini):
    spec, a = parse_yace(mini)
    assert spec.element_names == ["Si"] and a["Z"].tolist() == [14]
    assert a["crad"].shape == (1, 1, 1, 2, 2)
    np.testing.assert_array_equal(a["crad"][0, 0, 0], [[1, 0.5], [0.3, 0.2]])
    np.testing.assert_array_equal(a["radparams"][0, 0], [5.25, 5, 0.01, 0, 0])
    assert a["ctilde_complex"].shape == (4, 1)          # 1 + 3 terms
    assert [f[2:] for f in spec.functions] == [(0, 1, 1), (1, 3, 1)]
    np.testing.assert_array_equal(a["funcs"][1]["ms"], [[-1, 1], [0, 0], [1, -1]])


@pytest.mark.parametrize("edit, msg", [
    (("ChebExpCos", "ACE.jl.base"), "ACE.jl"),
    (("inner_cutoff_type: density", "inner_cutoff_type: zbl"), None),   # single bond: fine
])
def test_radbase_checks(tmp_path, edit, msg):
    p = tmp_path / "x.yace"
    p.write_text(MINI.replace(*edit))
    if msg is None:
        parse_yace(p)
    else:
        with pytest.raises(NotImplementedError, match=msg):
            parse_yace(p)


def test_mixed_bond_types_rejected(tmp_path, mini):
    t = load_tree(mini)
    from ace_jax.eval.pace_io import dump_tree
    t["elements"] = ["Si", "Ge"]; t["E0"] = [-1.5, -2.0]
    t["embeddings"][1] = dict(t["embeddings"][0])
    b = t["bonds"][(0, 0)]
    for k in [(0, 1), (1, 0), (1, 1)]:
        t["bonds"][k] = dict(b)
    t["functions"][1] = []
    t["bonds"][(1, 1)]["radbasename"] = "SBessel"
    p = tmp_path / "a.yace"; dump_tree(t, p)
    with pytest.raises(NotImplementedError, match="radbasename"):
        parse_yace(p)
    t["bonds"][(1, 1)]["radbasename"] = "ChebExpCos"
    t["bonds"][(1, 1)]["inner_cutoff_type"] = "distance"
    dump_tree(t, p)
    with pytest.raises(ValueError, match="inner_cutoff_type"):
        parse_yace(p)


@pytest.mark.parametrize("name", ["si_chebexpcos", "gesi_sbessel", "sige_zbl"])
def test_parse_fixtures(name):
    p = FIX / f"{name}.yace"
    pace_fixture(p)
    spec, a = parse_yace(p)
    NZ = len(spec.element_names)
    assert a["crad"].shape[:2] == (NZ, NZ)
    assert a["ctilde_complex"].shape[0] == spec.functions[-1][2] + spec.functions[-1][3]
    if name == "gesi_sbessel":
        assert a["Z"].tolist() == [32, 14]


def test_function_indices_beyond_bond_basis_rejected(tmp_path):
    """A function using l > lmax (or n > nradmax / nradbase) must fail at load,
    naming the problem, not deep inside the basis builder."""
    p = tmp_path / "bad.yace"
    p.write_text(MINI.replace("ls: [1, 1]", "ls: [2, 2]"))
    with pytest.raises(ValueError, match="exceeds"):
        parse_yace(p)


@pytest.mark.parametrize("edit, match", [
    (("inner_cutoff_type: density", "inner_cutoff_type: zbl2"), "inner_cutoff_type"),
    (("radbasename: ChebExpCos", "radbasename: ChebExpCos2"), "radbasename"),
])
def test_unknown_types_rejected_at_load(tmp_path, edit, match):
    """ML-PACE throws on these; they must fail at load, not evaluate as density."""
    p = tmp_path / "x.yace"
    p.write_text(MINI.replace(*edit))
    with pytest.raises(NotImplementedError, match=match):
        parse_yace(p)


def test_bond_defaults_match_mlpace_from_yaml(tmp_path):
    """ACEBondSpecification::from_YAML: absent rcut_in/dcut_in -> 0 (not the
    ACERadialFunctions::init 1e-5), and prehc/lambdahc are required."""
    p = tmp_path / "x.yace"
    p.write_text(MINI.replace(" rcut_in: 0, dcut_in: 0,", ""))
    _, a = parse_yace(p)
    np.testing.assert_array_equal(a["radparams"][0, 0, 3:], [0.0, 0.0])
    p.write_text(MINI.replace(" prehc: 0,", ""))
    with pytest.raises(ValueError, match="prehc"):
        parse_yace(p)
