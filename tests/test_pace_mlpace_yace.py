"""Multi-element linear ACE written by write_yace and evaluated by ML-PACE itself (its product and recursive C-tilde
evaluators, `pair_style pace product|recursive`, via pace_ref/tabulated/tab_eval) against ace-jax: two ACE1 models (SiGe, the 5-element Cantor) and two
species-embedded (tensor-reduced) models (SiGe, CrMnFe at d_max 4), each in the default form (every factor
expanded over the neighbour species its radial column is live for) and in the pooled form (`summed=True`,
`summed_neighbour_A`).

ML-PACE needs two additions for these files: `radbasename: Tabulated` (ICAMS/lammps-user-pace#31) for every
case, and `summed_neighbour_A` for the pooled ones, which a reader without it ignores and silently
mis-evaluates.  Both are on jameskermode/lammps-user-pace `tabulated-summed-fast`, the ML-PACE the pace-parity
job builds.  Against upstream main 99aa6e6 every case fails (the expanded ones at load, the pooled ones on the
values); with the tabulated commit alone the pooled ones fail.

ACEJAX_PACE_TAB_EVAL names the tab_eval binary (pace-parity builds it from pace_ref/CMakeLists.txt); without
it the tests skip, unless ACEJAX_REQUIRE_FIXTURES is set."""
import os
import subprocess

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from ase.build import bulk  # noqa: E402
from ase.data import atomic_numbers  # noqa: E402

import ace_jax as aj  # noqa: E402
from ace_jax.calc.point import ACECalculator  # noqa: E402
from ace_jax.eval import write_yace  # noqa: E402
from ace_jax.eval.pace_io import load_tree  # noqa: E402
from conftest import FIXTURE_DIR  # noqa: E402

# (fixture, table spacing): coarse spacings keep the NZ^2 bond tables small; the tolerance follows the spacing
MODELS = [("sige_nofit.npz", 5e-3), ("ace_cantor5_small.npz", 1e-2),
          ("emb_ref_SiGe_o2d6.npz", 5e-3), ("emb_ref_CrMnFe_o3d5_dmax4.npz", 5e-3)]
FTOL = {5e-3: 3e-6, 1e-2: 3e-5}           # cubic Hermite tables: O(h^3) in the derivatives (test_ace_yace)


def _tab_eval():
    exe = os.environ.get("ACEJAX_PACE_TAB_EVAL")
    if not exe or not os.path.exists(exe):
        if os.environ.get("ACEJAX_REQUIRE_FIXTURES"):
            pytest.fail(f"ACEJAX_PACE_TAB_EVAL={exe!r}: no tab_eval binary (pace_ref/CMakeLists.txt builds it)")
        pytest.skip("no ML-PACE tab_eval binary (set ACEJAX_PACE_TAB_EVAL)")
    return exe


def _cells(z, n=3, seed=0):
    """Rattled, mixed-species, partly strained diamond cells with every element present."""
    rng = np.random.default_rng(seed)
    out = []
    for c in range(n):
        at = bulk("Si", "diamond", 5.43 * (0.97 + 0.06 * rng.random()), cubic=True).repeat(2)
        at.numbers[:] = rng.choice(z, len(at))
        at.numbers[:len(z)] = z
        at.rattle(0.15, seed=int(rng.integers(1 << 30)))
        if c % 2:
            at.set_cell(at.cell @ (np.eye(3) + 0.03 * rng.standard_normal((3, 3))), scale_atoms=True)
        out.append(at)
    return out


def _run(exe, yace, ats, evaluator):
    """tab_eval's protocol (pace_ref/tabulated/tab_eval.cpp): (E, F) per configuration."""
    idx = {e: i for i, e in enumerate(load_tree(yace)["elements"])}
    inp = [str(len(ats))]
    for a in ats:
        inp += [str(len(a)), " ".join(map("{:.17g}".format, np.asarray(a.cell).ravel())),
                " ".join(str(int(p)) for p in a.pbc)]
        inp += [f"{idx[s]} " + " ".join(map("{:.17g}".format, p)) for s, p in zip(a.get_chemical_symbols(), a.positions)]
    r = subprocess.run([exe, str(yace), evaluator], input="\n".join(inp), capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"ML-PACE could not evaluate {yace.name}:\n{r.stderr[-2000:]}"
    lines, k, res = r.stdout.split("\n"), 0, []
    for a in ats:
        res.append((float(lines[k]), np.array([list(map(float, lines[k + 1 + i].split())) for i in range(len(a))])))
        k += 1 + len(a)
    return res


@pytest.mark.parametrize("evaluator", ["product", "recursive"])
@pytest.mark.parametrize("summed", [False, True], ids=["expanded", "summed"])
@pytest.mark.parametrize("name,delta", MODELS, ids=[m[0].split(".")[0] for m in MODELS])
def test_mlpace_evaluates_multi_element_export(tmp_path, name, delta, summed, evaluator):
    exe = _tab_eval()
    src = FIXTURE_DIR / name
    model, meta, _ = aj.load(src)
    yace = tmp_path / "m.yace"
    write_yace(model, yace, delta=delta, summed=summed)
    ats = _cells([atomic_numbers[e] for e in load_tree(yace)["elements"]])
    calc = ACECalculator(model, meta=meta, skin=0)
    ftol = FTOL[delta]
    for (E1, F1), a in zip(_run(exe, yace, ats, evaluator), ats):
        b = a.copy()
        b.calc = calc
        E0, F0 = b.get_potential_energy(), b.get_forces()
        assert abs(E1 - E0) / len(a) < 10 * ftol
        assert np.abs(F1 - F0).max() < ftol * max(1.0, np.abs(F0).max())
