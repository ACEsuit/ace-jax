"""Python-side model authoring: seeded radial init, build_model, npz bridge.

Two ladders:

* unit tests (numpy only, always run): the normalized-Legendre recurrence, the
  agnesi transform 7-tuple, and the envelope formulas, pinned against
  fixtures/si_ace_model.npz and numpy's own polynomials.
* bridge tests (need the `authoring` extra; skip cleanly otherwise): a Si
  order-3 totaldegree-10 model is authored from scratch, the committed Julia
  fixture's coefficients are injected (exact because the fixture's nnll row
  order fed back through `couple` reproduces its A2B bit-for-bit), packaged
  with `save_npz`, and compared array-for-array against the fixture, then
  evaluated against the fixture's energies, forces and descriptors.

juliacall note: standalone scripts need the repo root on PYTHONPATH so
juliapkg finds the repo's juliapkg.json (pytest provides it implicitly).
"""

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures")


# ---------------------------------------------------------------------------
#  unit tests (numpy only)
# ---------------------------------------------------------------------------

def _fixture(name):
    return np.load(os.path.join(FIX, name), allow_pickle=True)


def test_legendre_recurrence():
    from ace_jax.construct.radial_init import legendre_3term, poly_eval
    x = np.linspace(-1, 1, 41)
    n = 15
    A, B, C = legendre_3term(n)
    Q = poly_eval(x, A, B, C)
    import numpy.polynomial.legendre as L
    for k in range(n):
        c = math.sqrt((2 * k + 1) / 2)
        assert np.allclose(Q[:, k], c * L.legval(x, [0] * k + [1]), atol=1e-12)


def test_legendre_3term_matches_fixture():
    """The authored recurrence must be the exporter's, coefficient for
    coefficient: Julia's legendre_basis folds the P0 normalisation into A[2]
    (P1 = A[2] x + B[2], no P0 factor), so A[1] here is sqrt(3/2), not
    sqrt(3)."""
    from ace_jax.construct.radial_init import legendre_3term
    z = _fixture("si_ace_model.npz")
    A, B, C = legendre_3term(len(z["polys_A"]))
    assert np.allclose(A, z["polys_A"], atol=1e-12)
    assert np.allclose(B, z["polys_B"], atol=1e-12)
    assert np.allclose(C, z["polys_C"], atol=1e-12)


def test_poly_eval_matches_eval_path():
    """poly_eval (numpy, authoring side) and poly_recursion (JAX, eval side)
    read the same (A, B, C) with the same convention, so the unit test above
    pins what the eval path actually computes."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ace_jax.construct.radial_init import legendre_3term, poly_eval
    from ace_jax.eval.radial import poly_recursion
    x = np.linspace(-1, 1, 41)
    for n in (2, 3, 15):
        A, B, C = legendre_3term(n)
        Q = poly_eval(x, A, B, C)
        Qj = np.asarray(poly_recursion(jnp.asarray(x), jnp.asarray(A),
                                       jnp.asarray(B), jnp.asarray(C)))
        assert Q.shape == (41, n)
        assert np.allclose(Q, Qj, atol=1e-13)


def test_resolve_elements_rejects_duplicates():
    from ace_jax.construct.radial_init import resolve_elements
    assert resolve_elements(["C", "Si", 32]) == [6, 14, 32]     # order kept
    with pytest.raises(ValueError, match="duplicate"):
        resolve_elements(["Si", 14])


def test_per_pair_r0_default():
    """ACEpotentials _default_rin0cuts: r0(zi, zj) = (bond_len(zi) +
    bond_len(zj)) / 2 per species pair, not one global mean."""
    from ace_jax.construct.radial_init import (agnesi_transform_params,
                                               pair_radial_init,
                                               tensor_radial_init)
    Rnl = [(1, 0), (2, 0), (1, 1)]
    t = tensor_radial_init([14, 6], Rnl, rcut=5.5)
    r0 = t["rnl_transform"][..., 4]
    assert np.allclose(r0, [[2.4, 1.9], [1.9, 1.4]])
    # every slot is the full per-pair 7-tuple (ycut moves with r0)
    for i, j in ((0, 0), (0, 1), (1, 1)):
        ref = agnesi_transform_params(0.0, r0[i, j], 5.5, 2, 2)
        assert np.allclose(t["rnl_transform"][i, j], ref, atol=1e-12)
    pr = pair_radial_init([14, 6], 4, rcut=5.5)
    assert np.allclose(pr["pair_transform"][..., 4], [[2.4, 1.9], [1.9, 1.4]])


def test_r0_override_scalar_or_table():
    from ace_jax.construct.radial_init import tensor_radial_init
    Rnl = [(1, 0), (2, 0)]
    t = tensor_radial_init([14, 6], Rnl, rcut=5.5, r0=2.0)
    assert np.allclose(t["rnl_transform"][..., 4], 2.0)
    tab = np.array([[2.5, 2.0], [2.0, 1.5]])
    t = tensor_radial_init([14, 6], Rnl, rcut=5.5, r0=tab)
    assert np.allclose(t["rnl_transform"][..., 4], tab)
    with pytest.raises(ValueError):
        tensor_radial_init([14, 6], Rnl, rcut=5.5, r0=np.ones((3, 3)))


def test_untabulated_bond_length_raises():
    """Julia errors per element (`bond_len(z)`); a silent average over the
    known species would give a wrong transform for the unknown one."""
    from ace_jax.construct.radial_init import tensor_radial_init
    with pytest.raises(ValueError, match="bond length"):
        tensor_radial_init([14, 2], [(1, 0)], rcut=5.5)


def test_agnesi_params_match_fixture():
    from ace_jax.construct.radial_init import (agnesi_a, agnesi_transform_params,
                                               agnesi_normalized)
    z = _fixture("si_ace_model.npz")
    ref_t = np.asarray(z["rnl_transform"], float)[0, 0]
    tr = agnesi_transform_params(0.0, 2.4, 5.5, 2, 2)
    assert np.allclose(tr, ref_t, atol=1e-12)
    assert abs(agnesi_a(2, 2) - 2 / 3) < 1e-14
    # ycut must follow from the raw transform, not be stored independently
    from ace_jax.construct.radial_init import agnesi_raw
    assert abs(agnesi_raw(5.5, 2, 2, agnesi_a(2, 2), 0.0, 2.4) - ref_t[6]) < 1e-12
    # and the normalized eval agrees with the fixture's probe_x at probe_r
    x = agnesi_normalized(np.asarray(z["probe_r"], float), ref_t)
    assert np.allclose(x, np.asarray(z["probe_x"], float), atol=1e-12)


def test_envelope_formulas():
    from ace_jax.construct.radial_init import envelope1sr_eval, envelope2sx_eval
    x = np.linspace(-1, 1, 21)
    env = (-1.0, 1.0, 2.0, 2.0, 1.0)
    e = envelope2sx_eval(None, x, env)
    assert np.allclose(e, (1 - x**2)**2 * (x > -1) * (x < 1), atol=1e-12)
    r = np.linspace(0, 6.5, 131)
    ep = envelope1sr_eval(r, (5.5, 1.0))
    assert ep[r >= 5.5].max(initial=0.0) == 0.0
    # at r=0, t^-p diverges exactly as in Julia (Inf); physically rin > 0
    assert math.isinf(ep[0])
    assert np.isfinite(ep[(r > 0) & (r < 5.5)]).all()
    # envelope shape: rises from 0 at the inner limit, peaks, dies at rcut
    assert ep[1] > 0.0 and np.argmax(ep) < len(r) // 2


def test_eval_pair_in_memory_handoff():
    """A (model, meta) pair evaluates directly -- no npz round-trip.

    eval_pair's derived meta must equal the loader's for a fixture tree, and
    the pair path must agree with the path-based calculator (which test_efv
    separately pins against Julia).  Authoring is exercised via a directly
    instantiated NamedTuple, so this needs no Julia."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from ase import Atoms
    from ace_jax.eval import ACECalculator, load
    from ace_jax.construct.model import Authoring

    path = os.path.join(FIX, "si_ace_model.npz")
    z = _fixture("si_ace_model.npz")
    model, meta, _ = load(path)
    auth = Authoring(model=model, meta=meta, nnll_spec=(), Rnl_spec=(),
                     Ylm_spec=(), aa_sig=(), aspec=(), aa_specs=(), nnll=(),
                     gamma=None)
    _, meta2 = auth.eval_pair()
    for k in ("radial_kind", "pair_radial_kind", "pair_envelope_kind",
              "ybasis_kind", "elements", "n_B", "n_AA", "lmax", "rcut"):
        assert meta2[k] == meta[k], k
    # meta was deep-copied: mutating the copy must not leak into auth.meta
    meta2["rcut"] = -1.0
    assert auth.meta["rcut"] == meta["rcut"]

    atoms = Atoms(numbers=np.asarray(z["test_Z"]),
                  positions=np.asarray(z["test_pos"]).T,
                  cell=np.asarray(z["test_cell"]).T,
                  pbc=np.asarray(z["test_pbc"]).astype(bool))
    atoms.calc = ACECalculator(*auth.eval_pair())
    E2, F2, S2 = (atoms.get_potential_energy(), atoms.get_forces(),
                  atoms.get_stress(voigt=False))
    d2 = atoms.calc.get_site_descriptors(atoms)
    atoms.calc = ACECalculator(path)
    assert abs(E2 - atoms.get_potential_energy()) < 1e-12
    assert np.abs(F2 - atoms.get_forces()).max() < 1e-12
    assert np.abs(S2 - atoms.get_stress(voigt=False)).max() < 1e-12
    assert np.abs(d2 - atoms.calc.get_site_descriptors(atoms)).max() < 1e-12
    assert abs(E2 - float(z["test_E"][0])) < 1e-10


def _atoms_from(z):
    from ase import Atoms
    return Atoms(numbers=np.asarray(z["test_Z"]),
                 positions=np.asarray(z["test_pos"]).T,
                 cell=np.asarray(z["test_cell"]).T,
                 pbc=np.asarray(z["test_pbc"]).astype(bool))


def _blank_authoring(model, meta):
    from ace_jax.construct.model import Authoring
    return Authoring(model=model, meta=meta, nnll_spec=(), Rnl_spec=(),
                     Ylm_spec=(), aa_sig=(), aspec=(), aa_specs=(), nnll=(),
                     gamma=None)


def _fixture_coupling():
    """A `Coupling` reconstructed from fixtures/si_ace_model.npz.  The
    fixture's A2B columns are in aa_spec (evaluation) order, so each column's
    (n, l, m) signature comes from the fixture's own aspec gathers."""
    from ace_jax.construct.coupling import Coupling
    from ace_jax.construct.spec import build_spec
    z = _fixture("si_ace_model.npz")
    meta = json.loads(bytes(z["meta_json"]).decode())
    _, Rnl, Ylm = build_spec(1, 3, 10, 1.5)
    ar, ay = z["aspec_r"], z["aspec_y"]
    aa = [np.asarray(z[f"aa_spec_{k+1}"]) for k in range(len(meta["aa_lens"]))]
    sig = tuple(tuple((Rnl[ar[i]][0], Rnl[ar[i]][1], Ylm[ay[i]][1]) for i in row)
                for g in aa for row in g)
    return Coupling(A2B=np.asarray(z["A2B"], float), aa_sig=sig,
                    aspec=tuple((int(r), int(y)) for r, y in zip(ar, ay)),
                    aa_specs=tuple(aa),
                    nnll_spec=tuple(tuple(tuple(b) for b in bb) for bb in meta["nnll"]))


def _primed_cache(tmp_path):
    """A coupling cache dir holding the fixture coupling under the key
    build_model([14], 3, 10) asks for, so authoring never touches Julia."""
    from ace_jax.construct import coupling as C
    from ace_jax.construct.spec import build_spec
    mb, Rnl, Ylm = build_spec(1, 3, 10, 1.5)
    key = C.coupling_key(mb, Rnl, Ylm)
    C._write_entry(C._entry_path(tmp_path, key), _fixture_coupling(), key, mb, Rnl, Ylm)
    return str(tmp_path)


def test_build_model_edge_a_kind_matmul(tmp_path, monkeypatch):
    """edge_a_kind="matmul" needs the one-hot selectors the loader builds;
    an authored matmul model must carry them and evaluate identically to the
    gather form (WB is zero at authoring, so compare descriptors)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.construct.model import build_model
    from ace_jax.eval import ACECalculator
    monkeypatch.setenv("ACEJAX_NO_JULIA", "1")
    cache = _primed_cache(tmp_path)
    g = build_model([14], 3, 10, coupling_cache_dir=cache)
    m = build_model([14], 3, 10, coupling_cache_dir=cache, edge_a_kind="matmul")
    assert g.model.a_sel_r is None and m.model.edge_a_kind == "matmul"
    assert m.model.a_sel_r.shape == (g.meta["n_rnl"], g.meta["n_A"])
    assert m.model.a_sel_y.shape == (g.meta["n_ylm"], g.meta["n_A"])
    atoms = _atoms_from(_fixture("si_ace_model.npz"))
    dg = ACECalculator(*g.eval_pair()).get_site_descriptors(atoms)
    dm = ACECalculator(*m.eval_pair()).get_site_descriptors(atoms)
    assert np.isfinite(dg).all() and np.abs(dg).max() > 0
    assert np.abs(np.asarray(dg) - np.asarray(dm)).max() < 1e-12
    with pytest.raises(ValueError):
        build_model([14], 3, 10, coupling_cache_dir=cache, edge_a_kind="scatter")


def test_save_npz_spline_factorised_roundtrip(tmp_path):
    """A tree on the factorised-spline branch must save as the exporter's
    factorised layout and reload to the same energies (a trivial d=1
    factorisation of a spline fixture makes the reference exact)."""
    import dataclasses
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ace_jax.construct.export import save_npz
    from ace_jax.eval import ACECalculator, load

    path = os.path.join(FIX, "si_1429.npz")
    model, meta, _ = load(path)
    coefs = np.asarray(model.rnl_coefs)                       # (1, 1, ncoef, n_rnl)
    n_rnl = coefs.shape[-1]
    fac = dataclasses.replace(
        model, radial_kind="spline_factorised",
        rnl_coefs=jnp.zeros((1, 1, 1, 1), jnp.float64),
        rnl_coefs_single=jnp.asarray(coefs[0, 0]),
        rnl_embedding=jnp.ones((1, 1), jnp.float64),
        rnl_emb_nidx=jnp.arange(n_rnl, dtype=jnp.int32),
        rnl_emb_kidx=jnp.zeros(n_rnl, jnp.int32))
    out = str(tmp_path / "fac.npz")
    save_npz(out, _blank_authoring(fac, meta))
    m2, meta2, z2 = load(out)
    assert m2.radial_kind == "spline_factorised" and meta2["radial_kind"] == "spline_factorised"
    assert np.array_equal(np.asarray(m2.rnl_coefs_single), coefs[0, 0])
    assert np.array_equal(np.asarray(m2.rnl_emb_nidx), np.arange(n_rnl))
    assert meta2["rnl_spline"] == meta["rnl_spline"]
    atoms = _atoms_from(_fixture("si_1429.npz"))
    atoms.calc = ACECalculator(path)
    E, F = atoms.get_potential_energy(), atoms.get_forces()
    atoms.calc = ACECalculator(out)
    assert abs(atoms.get_potential_energy() - E) < 1e-10
    assert np.abs(atoms.get_forces() - F).max() < 1e-10


# ---------------------------------------------------------------------------
#  bridge tests (authoring extra)
# ---------------------------------------------------------------------------

_BRIDGE = r"""
import json, sys
import dataclasses

import numpy as np
import juliacall  # noqa: F401  (fail loudly here if the extra is missing)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ace_jax.construct.model import build_model
from ace_jax.construct.spec import build_spec
from ace_jax.construct.coupling import couple
from ace_jax.construct.export import save_npz
from ace_jax.eval import io as eio
from ace_jax.eval.model import fold_readout

auth = build_model([14], 3, 10)
m, meta = auth.model, auth.meta
zf = np.load(sys.argv[1], allow_pickle=True)
fmeta = json.loads(bytes(zf["meta_json"]).decode())
mb_ref = [[tuple(b) for b in bb] for bb in fmeta["nnll"]]

res = {}
res["nnll_multiset"] = (sorted(sorted(b) for b in auth.nnll_spec)
                        == sorted(sorted(b) for b in mb_ref))
# fixture nnll row order -> identical A2B (exact injection premise)
mb, Rnl, Ylm = build_spec(1, 3, 10, wL=1.5)
cpl = couple(mb_ref, Rnl, Ylm)
res["a2b_exact"] = bool(np.array_equal(cpl.A2B, np.asarray(zf["A2B"], float)))
res["shapes"] = {
    "A2B": list(m.A2B.shape), "WB": list(m.WB.shape), "Wpair": list(m.Wpair.shape),
    "rnl_Wnlq": list(m.rnl_Wnlq.shape), "pair_Wnlq": list(m.pair_Wnlq.shape),
}
res["meta"] = {k: meta[k] for k in
               ("aa_lens", "aa_orders", "lmax", "n_A", "n_AA", "n_B", "n_pair",
                "n_rnl", "n_ylm", "len_basis", "rcut")}

# inject the fixture's fitted coefficients and branch kinds, package, reload
A2B = np.asarray(cpl.A2B)
rr, cc = np.nonzero(A2B)
psp = fmeta["pair_spline"]
m2 = dataclasses.replace(m,
    A2B=jnp.asarray(A2B), a2b_rows=jnp.asarray(rr, jnp.int32),
    a2b_cols=jnp.asarray(cc, jnp.int32), a2b_vals=jnp.asarray(A2B[rr, cc]),
    rnl_Wnlq=jnp.asarray(zf["rnl_Wnlq"]),
    polys_A=jnp.asarray(zf["polys_A"]), polys_B=jnp.asarray(zf["polys_B"]),
    polys_C=jnp.asarray(zf["polys_C"]),
    rnl_transform=jnp.asarray(zf["rnl_transform"]),
    pair_transform=jnp.asarray(zf["pair_transform"]),
    rnl_envelope=jnp.asarray(zf["rnl_envelope"]),
    pair_envelope=jnp.asarray(zf["pair_envelope"]),
    pair_coefs=jnp.asarray(zf["pair_spline_coefs"]),
    pair_grid=(float(psp["x0"]), float(psp["h"]), int(psp["n"])),
    a2b_sparse=True, radial_kind="analytic", pair_radial_kind="spline",
    pair_envelope_kind="poly1sr",
    WB=jnp.asarray(zf["WB"]), Wpair=jnp.asarray(zf["Wpair"]),
    E0=jnp.asarray(zf["E0"]), folded=False)
# WB was replaced on a folded model: ctilde must be rebuilt (the stale-ctilde
# trap fold_readout's docstring warns about -- the round-trip below hid it,
# because load() rebuilds with folded=False, but the in-memory hand-off
# evaluates the tree as-is)
m2 = fold_readout(m2)
meta2 = dict(meta)
meta2["nnll"] = [[list(b) for b in bb] for bb in cpl.nnll_spec]
auth2 = auth._replace(model=m2, meta=meta2)
save_npz("/tmp/opencode_auth_roundtrip.npz", auth2)
model, rmeta, rz = eio.load("/tmp/opencode_auth_roundtrip.npz")

res["arrays"] = {}
for name, got, ref in (
        ("A2B", model.A2B, zf["A2B"]), ("WB", model.WB, zf["WB"]),
        ("Wpair", model.Wpair, zf["Wpair"]), ("E0", model.E0, zf["E0"]),
        ("rnl_Wnlq", model.rnl_Wnlq, zf["rnl_Wnlq"]),
        ("pair_spline_coefs", model.pair_coefs, zf["pair_spline_coefs"]),
        ("aspec_r", model.aspec_r, zf["aspec_r"]),
        ("aspec_y", model.aspec_y, zf["aspec_y"]),
        ("polys_A", model.polys_A, zf["polys_A"])):
    res["arrays"][name] = float(abs(np.asarray(got) - np.asarray(ref, float)).max())

# evaluate against the fixture's stored references (Julia calculator)
from ase import Atoms
from ace_jax.eval import ACECalculator

atoms = Atoms(numbers=zf["test_Z"], positions=zf["test_pos"].T,
              cell=zf["test_cell"].T, pbc=zf["test_pbc"].astype(bool))
atoms.calc = ACECalculator("/tmp/opencode_auth_roundtrip.npz")
E = float(atoms.get_potential_energy())
F = atoms.get_forces()
S = atoms.get_stress(voigt=False)
vol = atoms.get_volume()
d = np.asarray(atoms.calc.get_site_descriptors(atoms))
Vj = np.asarray(zf["test_V"], float)
res["efv"] = {
    "E": abs(E - float(zf["test_E"][0])),
    "F": float(np.abs(np.asarray(F) - zf["test_F"].T).max()),
    # ASE stress is -virial/volume; the fixture stores the Julia virial
    "S": float(np.abs(np.asarray(S) + Vj / vol).max()),
    "desc": float(np.abs(d - zf["test_desc"].T).max()),
}

# same tree, in memory (no file): eval_pair hand-off must match the round-trip
atoms.calc = ACECalculator(*auth2.eval_pair())
res["inmem"] = {
    "E": abs(float(atoms.get_potential_energy()) - float(zf["test_E"][0])),
    "F": float(np.abs(np.asarray(atoms.get_forces()) - zf["test_F"].T).max()),
    "S": float(np.abs(np.asarray(atoms.get_stress(voigt=False)) + Vj / vol).max()),
    "desc": float(np.abs(np.asarray(atoms.calc.get_site_descriptors(atoms))
                         - zf["test_desc"].T).max()),
}
print("RESULT", json.dumps(res))
"""


@pytest.mark.skipif(not os.path.exists(os.path.join(FIX, "si_ace_model.npz")),
                    reason="fixture missing")
def test_bridge_wellformed_subprocess():
    """Authored model is structurally identical to the committed Julia fixture,
    and with the fixture's coefficients injected it reproduces Julia's
    energies, forces, stress and descriptors through the eval path."""
    try:
        import juliacall  # noqa: F401
    except ImportError:
        pytest.skip("authoring extra (juliacall) not installed")
    p = subprocess.run([sys.executable, "-c", _BRIDGE,
                        os.path.join(FIX, "si_ace_model.npz")],
                       capture_output=True, text=True, timeout=1200)
    assert p.returncode == 0, f"bridge subprocess failed:\n{p.stderr[-2000:]}"
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")][-1]
    r = json.loads(line[len("RESULT "):])
    assert r["nnll_multiset"] and r["a2b_exact"]
    assert r["shapes"]["A2B"] == [110, 230]
    assert r["shapes"]["WB"] == [110, 1] and r["shapes"]["Wpair"] == [10, 1]
    assert r["shapes"]["rnl_Wnlq"] == [1, 1, 37, 15]
    assert r["shapes"]["pair_Wnlq"] == [1, 1, 10, 15]
    assert r["meta"]["aa_lens"] == [10, 81, 139]
    assert r["meta"]["lmax"] == 7 and r["meta"]["len_basis"] == 120
    assert r["meta"]["n_B"] == 110 and r["meta"]["n_AA"] == 230
    for name, err in r["arrays"].items():
        assert err < 1e-10, f"{name} round-trip error {err}"
    for stage in ("efv", "inmem"):
        for name, err in r[stage].items():
            assert err < 1e-8, f"{stage}/{name} parity error {err}"


_NOJULIA = r"""
import sys
import numpy as np
import juliacall  # noqa: F401  (fail loudly here if the extra is missing)
import jax
jax.config.update("jax_enable_x64", True)
from ace_jax.construct.model import build_model

auth = build_model([14], 2, 5, coupling_cache=True, coupling_cache_dir=sys.argv[1])
print("RESULT", float(np.asarray(auth.model.A2B).sum()), len(auth.meta["nnll"]))
"""


def test_coupling_cache_no_julia_on_hit(tmp_path):
    """Tier 2 point 2: run 1 populates the per-shape cache (Julia allowed);
    run 2 repeats with ACEJAX_NO_JULIA=1, which makes the shim raise if Julia
    is ever touched -- the build must be served entirely from cache and
    produce the same A2B."""
    try:
        import juliacall  # noqa: F401
    except ImportError:
        pytest.skip("authoring extra (juliacall) not installed")
    p1 = subprocess.run([sys.executable, "-c", _NOJULIA, str(tmp_path)],
                        capture_output=True, text=True, timeout=1200)
    assert p1.returncode == 0, f"cache-populating run failed:\n{p1.stderr[-2000:]}"
    p2 = subprocess.run([sys.executable, "-c", _NOJULIA, str(tmp_path)],
                        capture_output=True, text=True, timeout=1200,
                        env=dict(os.environ, ACEJAX_NO_JULIA="1"))
    assert p2.returncode == 0, (f"cache-hit run failed (missed the cache, or "
                                f"touched Julia):\n{p2.stderr[-2000:]}")
    assert p1.stdout.split("RESULT ")[-1] == p2.stdout.split("RESULT ")[-1]
