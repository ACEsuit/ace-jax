"""Fit on the re-centred Cantor defect benchmark (bench365) with the ace-jax pipeline API and
write the run layout (per-atom pred_*.npz) + CLI layout (metrics.csv, model file).

    python fit_bench.py <data_dir> <out_dir> pops|gp|ard [--smoke]
"""
import sys

import jax

jax.config.update("jax_enable_x64", True)

from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data, write_outputs   # noqa: E402
from ace_jax.fit.pipeline.outputs import checkpoint_writer                      # noqa: E402

data, out, arm = sys.argv[1], sys.argv[2], sys.argv[3]
smoke = "--smoke" in sys.argv
common = dict(model="/data/cantor_embed_d16_deg10.npz" if data.startswith("/data") else f"{data}/cantor_embed_d16_deg10.npz", energy_key="mace_energy", force_key="mace_force",
              virial_key="mace_virial", r0=2.5, batch=4, rungs=("map",), predict_train=False, e0="lsq",
              opt="lbfgs", map_steps=3 if smoke else 40, predict_stats="recompute")
if arm == "pops":
    cfg = FitConfig(**common, arm="linear", uq="pops", pops_ridge="auto")
elif arm == "ard":      # tempered ARD posterior (feat/ard-uq): joint type-II ML, kappa from a 20 % train hold-out
    cfg = FitConfig(**common, arm="linear", uq="ard", ard_mode="joint", ard_val_frac=0.2, ard_laplace=True)
else:
    cfg = FitConfig(**common, arm="gp", m_per_species=100, density="pca", pca_d=128, lml="host-cache",
                    map_restarts=1 if smoke else 4)
cfg.validate()
d = load_fit_data(cfg, train=f"{data}/train.xyz", test=f"{data}/test.xyz", ood=f"{data}/ood.xyz")
res = fit(cfg, d, log=lambda *s: print(*s, flush=True), on_stage=checkpoint_writer(out))   # stages survive a later failure
write_outputs(res, out, layout=("run", "cli"), argv={"arm": arm, "data": data})
print("done", {k: round(v, 1) for k, v in res.timings.items()}, flush=True)
