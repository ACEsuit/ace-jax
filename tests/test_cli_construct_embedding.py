import json, pathlib
import numpy as np
import pytest

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"


def _restamped_cache(tmp):
    from ace_jax.construct import coupling as C
    out = tmp / "cpl"
    for f in sorted((FIX / "coupling_cache_embedding").glob("cpl-*.npz")):
        z = np.load(f); m = json.loads(bytes(z["meta_json"]).decode())
        n = sum(1 for k in z.files if k.startswith("aa_spec_"))
        cpl = C.Coupling(A2B=np.asarray(z["A2B"]),
                         aa_sig=tuple(tuple(tuple(int(v) for v in t) for t in s) for s in m["aa_sig"]),
                         aspec=tuple((int(r), int(y)) for r, y in m["aspec"]),
                         aa_specs=tuple(np.asarray(z[f"aa_spec_{k + 1}"]) for k in range(n)),
                         nnll_spec=tuple(tuple((int(b[0]), int(b[1])) for b in bb) for bb in m["nnll_spec"]))
        C._write_entry(C._entry_path(out, m["key"]), cpl, m["key"], [[tuple(b) for b in bb] for bb in m["mb"]],
                       [tuple(r) for r in m["rnl"]], [tuple(y) for y in m["ylm"]])
    return out


def test_construct_embedding_matches_the_library(tmp_path, monkeypatch):
    import jax
    jax.config.update("jax_enable_x64", True)
    monkeypatch.setenv("ACEJAX_NO_JULIA", "1")
    from ace_jax.cli import main
    from ace_jax.construct.model import build_embedding_model
    from ace_jax.eval import load
    z = np.load(FIX / "embedding_ref_mh1_SiGe.npz")
    table = tmp_path / "emb.json"
    table.write_text(json.dumps({"Z": [14, 32], "emb": z["table"].tolist()}))
    cache = _restamped_cache(tmp_path)
    out = tmp_path / "m.npz"
    main(["construct", "--elements", "Si,Ge", "--order", "2", "--max-degree", "6", "--embedding", str(table),
          "--maxl", "6", "--coupling-cache-dir", str(cache), "--out", str(out)])
    model, meta, _ = load(out)
    ref = build_embedding_model([14, 32], 2, 6, embedding=([14, 32], z["table"]), maxl=6,
                                coupling_cache_dir=str(cache))
    for k in ("n_B", "n_pair", "n_rnl", "len_basis"):
        assert meta[k] == ref.meta[k], k
    assert json.loads(meta["embedding"])["widths"] == [2, 3]
