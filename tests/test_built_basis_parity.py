"""Design-matrix parity of a basis BUILT in Python against ACEpotentials.

test_gp_rows.py holds ace-jax's rows to ACEfit's design matrix, but for a model
ACEpotentials built and coupled (si_fitted.npz): it tests the evaluator, not the
builder.  Here the basis comes from `build_model` with the LIVE coupling library
(no cache, no fixture coupling), and only the radials are taken from the
reference (julia/built_basis_reference.jl), so the spec, aspec, aa_specs and
A2B -- including the column order #57 got wrong -- are what is compared.

Comparison, per centre species and per nnll body: the Python design columns of
that body must span the same space as the Julia ones.  ET main (the coupling
library) rescales B rows against the ET release ACEpotentials pins, and at
order >= 4 a block of multiplicity m is defined only up to an m x m rotation, so
each side must be a linear combination of the other's block, to 1e-9 relative.
The pair columns, the B-row bodies and the smoothness prior must match exactly.
"""
import json
import os

import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR, require_coupling_lib

jax.config.update("jax_enable_x64", True)

# julia/built_basis_reference.jl CASES; named, not globbed, so a regeneration
# that wrote nothing fails under ACEJAX_REQUIRE_FIXTURES instead of collecting none
CASES = ["Si_o3d10", "Si_o4d12", "SiGe_o3d6"]
TOL = 1e-9


def _design(z, n_cfg, L):
    """Reference rows [E; F (atom-major xyz); V (Voigt)] per config, unweighted."""
    from ace_jax.fit.data import VOIGT
    return np.concatenate([np.concatenate([z[f"cfg{k}_E"][None], z[f"cfg{k}_F"].reshape(-1, L),
                                           np.stack([z[f"cfg{k}_V"][a, b] for a, b in VOIGT])])
                           for k in range(1, n_cfg + 1)])


def _built_rows(z, meta):
    """The same rows from a Python-built basis carrying the reference radials."""
    import dataclasses

    import jax.numpy as jnp
    from ase import Atoms
    from ace_jax.basis.model import build_model
    from ace_jax.eval import highest_precision
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.rows import linear_rows
    b = build_model(meta["elements"], meta["order"], meta["totaldegree"], wL=meta["wL"],
                    rcut=meta["rcut"], coupling_cache=False)
    assert [tuple(r) for r in z["rnl_spec"].tolist()] == list(b.Rnl_spec), "Rnl spec differs: radials cannot be grafted"
    sp = meta["pair_spline"]
    model = dataclasses.replace(
        b.model, **{k: jnp.asarray(z[k]) for k in ("rnl_Wnlq", "polys_A", "polys_B", "polys_C", "rnl_transform",
                                                   "rnl_envelope", "pair_transform", "pair_envelope")},
        pair_coefs=jnp.asarray(z["pair_spline_coefs"]), pair_grid=(float(sp["x0"]), float(sp["h"]), int(sp["n"])),
        pair_radial_kind="spline", pair_envelope_kind="poly1sr")
    model, bmeta = b._replace(model=model).eval_pair()
    n_cfg, NZ = meta["n_configs"], len(meta["elements"])
    atoms = []
    for k in range(1, n_cfg + 1):
        n = len(z[f"cfg{k}_Z"])
        a = Atoms(numbers=z[f"cfg{k}_Z"], positions=z[f"cfg{k}_pos"].T, cell=z[f"cfg{k}_cell"].T, pbc=True)
        a.info["energy"], a.info["virial"], a.arrays["forces"] = 0.0, np.zeros((3, 3)), np.zeros((n, 3))
        atoms.append(a)
    configs = load_configs(atoms)
    ds = build_dataset(configs, bmeta, np.zeros(NZ), configs_per_batch=n_cfg)
    cfg = GPConfig(r0=2.35, rcut=float(bmeta["rcut"]), n_B=bmeta["n_B"], n_pair=bmeta["n_pair"], NZ=NZ, C=n_cfg)
    L = cfg.len_basis
    batch = jax.tree.map(lambda a: a[0], ds)
    with highest_precision():
        rows = linear_rows(model, cfg, batch)[0]
    out = []
    for c in range(n_cfg):
        nodes = np.flatnonzero(np.asarray(batch.node_cfg) == c)
        out += [np.asarray(rows.E[c, :L])[None], np.asarray(rows.F[nodes, :, :L]).reshape(-1, L),
                np.asarray(rows.V[c, :, :L])]
    return b, np.concatenate(out)


def _span_residual(P, J):
    """Max relative residual of each side's columns fitted by the other's span."""
    fit = lambda X, Y: np.abs(X - Y @ np.linalg.lstsq(Y, X, rcond=None)[0]).max() / np.abs(X).max()
    return max(fit(P, J), fit(J, P))


@pytest.mark.parametrize("case", CASES)
def test_built_basis_design_matrix_matches_acepotentials(case):
    ref = FIXTURE_DIR / f"built_basis_ref_{case}.npz"
    if not ref.exists():
        (pytest.fail if os.environ.get("ACEJAX_REQUIRE_FIXTURES") else pytest.skip)(
            f"missing {ref.name}; see julia/built_basis_reference.jl")
    require_coupling_lib()
    z = np.load(ref)
    meta = json.loads(bytes(z["meta_json"]).decode())
    b, P = _built_rows(z, meta)
    nB, nP, NZ, L = meta["n_B"], meta["n_pair"], len(meta["elements"]), meta["len_basis"]
    J = _design(z, meta["n_configs"], L)
    assert P.shape == J.shape == (J.shape[0], (nB + nP) * NZ)
    key = lambda bb: tuple(sorted(tuple(x) for x in bb))
    jrows, prows = {}, {}
    for r, bb in enumerate(meta["nnll"]):
        jrows.setdefault(key(bb), []).append(r)
    for r, bb in enumerate(b.nnll):
        prows.setdefault(key(bb), []).append(r)
    assert {k: len(v) for k, v in prows.items()} == {k: len(v) for k, v in jrows.items()}, "nnll bodies differ"
    worst, where = 0.0, None
    for zc in range(NZ):
        for body, jr in jrows.items():
            pc, jc = [zc * nB + r for r in prows[body]], [zc * nB + r for r in jr]
            if not np.abs(J[:, jc]).max() > 0:                     # body never seen by this species' sites
                continue
            res = _span_residual(P[:, pc], J[:, jc])
            if res > worst:
                worst, where = res, (zc, body, len(jr))
    assert worst < TOL, f"B columns do not span ACEpotentials' block: residual {worst:.3g} at (species, body, mult) {where}"
    pair = slice(NZ * nB, None)
    assert np.abs(P[:, pair] - J[:, pair]).max() < TOL * np.abs(J[:, pair]).max()
    # the prior names the body each row evaluates (#57 attached it to the wrong rows)
    gj = {key(bb): g for bb, g in zip(meta["nnll"], z["gamma"][:nB])}
    np.testing.assert_allclose(b.gamma[:nB], [gj[key(bb)] for bb in b.nnll], rtol=1e-12)
