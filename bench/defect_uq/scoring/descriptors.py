"""Per-site descriptors (production basis, compact D = 3007) for the bench365 train / test / ood
sets, in the same site order as the pipeline's pred_*.npz rows (config-major, live sites), plus
per-site species, family, config id, fixed-boundary and r_core (big cells).  -> sites.npz

    python descriptors.py
"""
import pathlib
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
from ase.io import read                                       # noqa: E402
from ace_jax.fit.inducing import GPConfig, site_features      # noqa: E402
from ace_jax.fit.pipeline import FitConfig, load_fit_data     # noqa: E402

B = pathlib.Path.home() / "acegp-data" / "cantor" / "bench365"
HERE = pathlib.Path(__file__).parent
cfg = FitConfig(model=str(B.parent / "cantor_embed_d16_deg10.npz"), arm="linear", r0=2.5, batch=4,
                energy_key="mace_energy", force_key="mace_force", virial_key="mace_virial", e0="model")
d = load_fit_data(cfg, train=str(B / "train.xyz"), test=str(B / "test.xyz"), ood=str(B / "ood.xyz"))
meta = d.meta
g = GPConfig(r0=2.5, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"], NZ=len(meta["elements"]), C=4)
out = {"order": np.array([len(x) for x in meta["nnll"]]), "n_B": meta["n_B"]}
from ace_jax.fit.data import build_dataset, load_configs          # noqa: E402
big = build_dataset(load_configs(str(B / "big.xyz"), "mace_energy", "mace_force", "mace_virial"),
                    meta, d.E0, 1)                                 # one cell per batch: no padding to 4k atoms
for name, ds in (("train", d.ds_train), ("test", d.ds_test), ("ood", d.ds_ood), ("big", big)):
    t = time.time()
    X, _ = site_features(d.model, g, ds)
    m = np.asarray(ds.node_mask).reshape(-1)
    out[f"X_{name}"] = np.asarray(X).reshape(-1, X.shape[-1])[m].astype(np.float32)
    out[f"Z_{name}"] = np.asarray(ds.node_z).reshape(-1)[m]
    ats = read(B / f"{name}.xyz", ":")
    out[f"fam_{name}"] = np.concatenate([[a.info["family"]] * len(a) for a in ats])
    out[f"cfg_{name}"] = np.concatenate([[k] * len(a) for k, a in enumerate(ats)])
    out[f"fixed_{name}"] = np.concatenate([a.arrays.get("fixed", np.zeros(len(a), bool)).astype(bool) for a in ats])
    out[f"rcore_{name}"] = np.concatenate([a.arrays.get("r_core", np.full(len(a), np.nan)) for a in ats])
    assert len(out[f"fam_{name}"]) == len(out[f"X_{name}"]), (name, "site order mismatch")
    print(name, out[f"X_{name}"].shape, f"{time.time() - t:.0f}s", flush=True)
np.savez(HERE / "sites.npz", **out)
