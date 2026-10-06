"""Fit on the re-centred Cantor defect benchmark (bench365) with the ace-jax pipeline API and
write the run layout (per-atom pred_*.npz) + CLI layout (metrics.csv, model file).

    python fit_bench.py <data_dir> <out_dir> pops|gp|ard|ard_<tag>|<ARD_ARMS name> [--smoke] [--train-extra A.xyz,B.xyz]

ARD_ARMS (ard_arms.py) are the ablation / sweep arms of the conformal force-sigma validation; --train-extra
appends those xyz configurations to train.xyz before the fit (crack realisations not held out, for the ell sweep).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)

from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data, write_outputs   # noqa: E402
from ace_jax.fit.pipeline.outputs import checkpoint_writer                      # noqa: E402

data, out, arm = sys.argv[1], sys.argv[2], sys.argv[3]
smoke = "--smoke" in sys.argv
extra = sys.argv[sys.argv.index("--train-extra") + 1].split(",") if "--train-extra" in sys.argv else []
common = dict(model="/data/cantor_embed_d16_deg10.npz" if data.startswith("/data") else f"{data}/cantor_embed_d16_deg10.npz", energy_key="mace_energy", force_key="mace_force",
              virial_key="mace_virial", r0=2.5, batch=4, rungs=("map",), predict_train=False, e0="lsq",
              opt="lbfgs", map_steps=3 if smoke else 40, predict_stats="recompute")
from ard_arms import make_config   # noqa: E402

# ARD arms pin e0="prefit": every recorded ARD run (before 2026-10-03, incl. the rev-2 acceptance tables)
# fitted with the pre-fit E0; since 2026-10-03 the library fits E0 jointly under e0="lsq" (as BLR)
ard_common = dict(common, e0="prefit")
if (arm_cfg := make_config(arm, ard_common)) is not None:
    cfg = arm_cfg
elif arm == "pops":
    cfg = FitConfig(**common, arm="linear", uq="pops", pops_ridge="auto")
elif arm.startswith("ard"):   # "ard", or "ard_<tag>" for a separate output dir      # tempered ARD posterior (feat/ard-uq): joint type-II ML, kappa from a 20 % train hold-out
    # "ard_c<k>": ard_cond_max = 10**k (the conditioning floor on the prior precisions); default 1e14
    cm = float(10 ** int(arm.split("_c")[1])) if "_c" in arm else 1e14
    cfg = FitConfig(**ard_common, arm="linear", uq="ard", ard_mode="joint", ard_val_frac=0.2, ard_laplace=True,
                    ard_cond_max=cm, ard_force_shape="iso")   # recorded `ard` / `ard_c<k>` results are iso (library default became aniso 2026-10-03)
elif arm == "gp_conv":
    # GP-discrepancy Phase 0 follow-up: the bench365 GP's MAP stopped at map_steps = 40 in every restart, with rho
    # on its upper bound; this refit runs L-BFGS to convergence (300 iterations per start) on the #60 fast path
    # (PCA-16 + host-cache), and saves gp_model.npz
    cfg = FitConfig(**{**common, "map_steps": 3 if smoke else 300}, arm="gp", m_per_species=100, density="pca",
                    pca_d=16, lml="host-cache", map_restarts=1 if smoke else 4)
else:
    cfg = FitConfig(**common, arm="gp", m_per_species=100, density="pca", pca_d=128, lml="host-cache",
                    map_restarts=1 if smoke else 4)
cfg.validate()
train = f"{data}/train.xyz"
if extra:   # train.xyz + the validated extra configurations in one extxyz (see train_extra.py)
    from train_extra import prepare
    os.makedirs(out, exist_ok=True)
    train = f"{out}/train_plus_extra.xyz"
    prepare(f"{data}/train.xyz", extra, train, log=lambda *s: print(*s, flush=True))
d = load_fit_data(cfg, train=train, test=f"{data}/test.xyz", ood=f"{data}/ood.xyz")
res = fit(cfg, d, log=lambda *s: print(*s, flush=True), on_stage=checkpoint_writer(out))   # stages survive a later failure
write_outputs(res, out, layout=("run", "cli"), argv={"arm": arm, "data": data, "train_extra": extra})
print("done", {k: round(v, 1) for k, v in res.timings.items()}, flush=True)
