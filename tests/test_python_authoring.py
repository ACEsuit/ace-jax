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
