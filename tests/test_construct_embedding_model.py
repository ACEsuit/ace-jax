"""Python-authored embedded model (build_embedding_model) against Julia's
ace_embedding_model (julia/embedding_desc_reference.jl, julia/export_model.jl):

* descriptor-subspace parity per centre species on hundreds of random-structure
  sites -- the same function space, whatever the basis order;
* basis sizes and the smoothness prior (tensor block on the UNFOLDED n');
* lossless widths are independent of the PCA sign convention (built from the
  port's own reduction, not Julia's rows);
* the npz bridge round-trips.

Couplings come from the committed cache fixtures/coupling_cache_embedding (the ET
shim ran once to fill it), so the tests are pip-only."""
import json
import pathlib

import numpy as np
import pytest

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
CACHE = None                                  # set by the _cache fixture (a restamped copy)
CASES = {"SiGe_o2d6": "emb_ref_SiGe_o2d6", "CrMnFe_o3d5_dmax4": "emb_ref_CrMnFe_o3d5_dmax4"}


def _need(p):
    if not p.exists():
        pytest.skip(f"missing {p.name}; see julia/embedding_desc_reference.jl / export_model.jl")
    return np.load(p)


@pytest.fixture(scope="module", autouse=True)
def _x64():
    import jax
    jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", autouse=True)
def _cache(tmp_path_factory):
    """The committed coupling entries, re-written into a temp cache by this
    process (as test_python_authoring's _primed_cache does), so their pin stamp
    is this environment's juliapkg hash -- which depends on where juliapkg.json
    sits on sys.path, not on the coupling.  A stale coupling would still fail
    the descriptor parity against Julia."""
    import json as _json
    from ace_jax.construct import coupling as C
    global CACHE
    out = tmp_path_factory.mktemp("cpl")
    for f in sorted((FIX / "coupling_cache_embedding").glob("cpl-*.npz")):
        z = np.load(f)
        m = _json.loads(bytes(z["meta_json"]).decode())
        n_orders = sum(1 for k in z.files if k.startswith("aa_spec_"))
        cpl = C.Coupling(
            A2B=np.asarray(z["A2B"]),
            aa_sig=tuple(tuple(tuple(int(v) for v in t) for t in sig) for sig in m["aa_sig"]),
            aspec=tuple((int(r), int(y)) for r, y in m["aspec"]),
            aa_specs=tuple(np.asarray(z[f"aa_spec_{k + 1}"]) for k in range(n_orders)),
            nnll_spec=tuple(tuple((int(b[0]), int(b[1])) for b in bb) for bb in m["nnll_spec"]))
        mb = [[tuple(b) for b in bb] for bb in m["mb"]]
        rnl, ylm = [tuple(r) for r in m["rnl"]], [tuple(y) for y in m["ylm"]]
        C._write_entry(C._entry_path(out, m["key"]), cpl, m["key"], mb, rnl, ylm)
    CACHE = str(out)


def _build(case, rows=True, **kw):
    from ace_jax.construct.model import build_embedding_model
    z = _need(FIX / f"embdesc_ref_{case}.npz")
    dmax = int(z["dmax"])
    src = dict(rows=z["rows"]) if rows else kw.pop("src")
    return build_embedding_model([int(x) for x in z["zlist"]], int(z["order"]), int(z["totaldegree"]),
                                 d_max=None if dmax == 0 else dmax, maxl=6,
                                 coupling_cache_dir=CACHE, **src, **kw), z


def _descs(auth, z):
    from ace_jax.eval.api import site_descriptors
    model, meta = auth.eval_pair()
    out, a = [], 0
    for n in z["nat"]:
        n = int(n)
        out.append(site_descriptors(model, z["pos"][a:a + n], z["Z"][a:a + n], cell=z["cells"][3 * len(out):3 * len(out) + 3],
                                    pbc=True, meta=meta))
        a += n
    return np.concatenate(out)                                        # (n_sites, len_basis)


def _span_residual(A, B, rtol=1e-9):
    """Relative residual of B's columns outside span(A), and the two ranks."""
    def orth(M):
        U, s, _ = np.linalg.svd(M, full_matrices=False)
        return U[:, s > rtol * s[0]]
    QA, QB = orth(A), orth(B)
    res = np.linalg.norm(B - QA @ (QA.T @ B)) / np.linalg.norm(B)
    return res, QA.shape[1], QB.shape[1]


@pytest.mark.parametrize("case", CASES)
def test_descriptor_subspace_matches_julia(case, monkeypatch):
    monkeypatch.setenv("ACEJAX_NO_JULIA", "1")
    auth, z = _build(case)
    Dp, Dj = _descs(auth, z), z["desc"].T
    assert Dp.shape == Dj.shape
    for s, zz in enumerate(z["zlist"]):
        sites = z["Z"] == zz
        r1, ra, rb = _span_residual(Dp[sites], Dj[sites])
        r2, _, _ = _span_residual(Dj[sites], Dp[sites])
        assert ra == rb and ra > 0, (int(zz), ra, rb)
        assert r1 < 1e-8 and r2 < 1e-8, (int(zz), r1, r2)


@pytest.mark.parametrize("case", CASES)
def test_basis_sizes_and_prior_match_julia(case, monkeypatch):
    monkeypatch.setenv("ACEJAX_NO_JULIA", "1")
    auth, z = _build(case)
    ref = _need(FIX / f"{CASES[case]}.npz")
    rm = json.loads(bytes(ref["meta_json"]).decode())
    for k in ("n_B", "n_pair", "n_rnl", "len_basis"):
        assert auth.meta[k] == rm[k], k
    assert json.loads(auth.meta["embedding"])["widths"] == [int(w) for w in z["widths"]]
    assert np.allclose(np.sort(auth.gamma), np.sort(z["gamma"]), rtol=1e-12)


def test_lossless_model_is_independent_of_the_pca_sign(monkeypatch):
    """Lossless widths: built from the port's own (sign-fixed) reduction of the
    MH-1 table, the model spans the same space as Julia's."""
    monkeypatch.setenv("ACEJAX_NO_JULIA", "1")
    t = _need(FIX / "embedding_ref_mh1_SiGe.npz")
    auth, z = _build("SiGe_o2d6", rows=False, src=dict(embedding=([14, 32], t["table"])))
    Dp, Dj = _descs(auth, z), z["desc"].T
    for zz in z["zlist"]:
        sites = z["Z"] == zz
        assert _span_residual(Dj[sites], Dp[sites])[0] < 1e-8


def test_save_npz_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("ACEJAX_NO_JULIA", "1")
    from ace_jax.construct.export import save_npz
    from ace_jax.eval import load
    from ace_jax.eval.api import site_descriptors
    auth, z = _build("SiGe_o2d6")
    save_npz(tmp_path / "m.npz", auth)
    model, meta, _ = load(tmp_path / "m.npz")
    n = int(z["nat"][0])
    a = site_descriptors(model, z["pos"][:n], z["Z"][:n], cell=z["cells"][:3], pbc=True, meta=meta)
    m_mem, meta_mem = auth.eval_pair()
    b = site_descriptors(m_mem, z["pos"][:n], z["Z"][:n], cell=z["cells"][:3], pbc=True, meta=meta_mem)
    assert meta["radial_kind"] in ("spline", "spline_factorised")
    assert np.abs(np.asarray(a) - np.asarray(b)).max() < 1e-13
