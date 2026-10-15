"""Dump raw linear design rows (E, F, V) and site descriptors X for a subsample of a training set,
computed in float32 (x64 off) or float64, so `analyse.py` can compare them (plan step 1:
docs/dev/plans/2026-10-15-fp32-gpus.md).

The precision is chosen per process, before jax is imported: with x64 off every array the
library builds is float32, which is what an fp32 fit would see.  The Dataset is built in numpy
float64 and cast on transfer, so rij is formed in fp64 and then rounded (the plan's rule).

    uv run --no-sync python bench/fp32/rows_dump.py --prec 32 --model M.npz --train T.xyz --out o.npz
"""
import argparse
import json
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--prec", choices=["32", "64"], required=True)
p.add_argument("--matmul", default="highest", help="jax_default_matmul_precision: highest | default (TF32)")
p.add_argument("--model", required=True)
p.add_argument("--train", required=True)
p.add_argument("--weights", default=None)
p.add_argument("--stride", type=int, default=8, help="keep every stride-th config")
p.add_argument("--per-batch", type=int, default=2)
p.add_argument("--max-atoms", type=int, default=256, help="drop configs larger than this (padding)")
p.add_argument("--out", required=True)
a = p.parse_args()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", a.prec == "64")
if a.matmul != "default":
    jax.config.update("jax_default_matmul_precision", a.matmul)

import equinox as eqx  # noqa: E402
import numpy as np  # noqa: E402

from ace_jax.eval import load  # noqa: E402
from ace_jax.fit.data import build_dataset, load_configs  # noqa: E402
from ace_jax.fit.inducing import GPConfig  # noqa: E402
from ace_jax.fit.rows import linear_rows  # noqa: E402

model, meta, z = load(a.model)
cfgs = load_configs(a.train, "energy", "forces", "virial", weights=a.weights and json.load(open(a.weights)))
cfgs = [c for c in cfgs[::a.stride] if len(c.numbers) <= a.max_atoms]
E0 = np.asarray(z["E0"], np.float64)
ds = build_dataset(cfgs, meta, E0, a.per_batch, pack=False, log=lambda *_: None)
g = GPConfig(r0=2.4, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
             NZ=len(meta["elements"]), C=int(ds.y_E.shape[1]))
rows = eqx.filter_jit(lambda m, b: linear_rows(m, g, b)[:2])

fdt = np.float64 if a.prec == "64" else np.float32
E, F, V, X, yE, yF, yV, wE, wF, wV, nat, Xcfg = ([] for _ in range(12))
times = []
for i in range(ds.n_batches):
    b = jax.tree.map(lambda x: x[i], ds)
    t = time.perf_counter()
    r, Xi = rows(model, b)
    jax.block_until_ready((r, Xi))
    times.append(time.perf_counter() - t)
    cm, nm = np.asarray(b.cfg_mask), np.asarray(b.node_mask)
    E.append(np.asarray(r.E)[cm]); V.append(np.asarray(r.V)[cm].reshape(-1, r.V.shape[-1]))
    F.append(np.asarray(r.F)[nm].reshape(-1, r.F.shape[-1])); X.append(np.asarray(Xi)[nm])
    Xcfg.append(np.asarray(b.node_cfg)[nm] + sum(len(e) for e in E[:-1]))   # global config index
    yE.append(np.asarray(b.y_E, np.float64)[cm]); wE.append(np.asarray(b.w_E, np.float64)[cm])
    yV.append(np.asarray(b.y_V, np.float64)[cm].ravel()); wV.append(np.repeat(np.asarray(b.w_V, np.float64)[cm], 6))
    yF.append(np.asarray(b.y_F, np.float64)[nm].ravel()); wF.append(np.repeat(np.asarray(b.w_F, np.float64)[nm], 3))
    nat.append(np.asarray(b.n_atoms)[cm])
cat = np.concatenate
info = dict(prec=a.prec, matmul=a.matmul, device=str(jax.devices()[0]), model=a.model, n_configs=len(cfgs),
            n_batches=int(ds.n_batches), batch_shape=list(ds.nbr.shape[1:]), L=int(E[0].shape[1]),
            t_first=times[0], t_batch_median=float(np.median(times[1:])) if len(times) > 1 else None,
            dtype=str(E[0].dtype))
np.savez(a.out, E=cat(E).astype(fdt), F=cat(F).astype(fdt), V=cat(V).astype(fdt), X=cat(X).astype(fdt),
         Xcfg=cat(Xcfg), yE=cat(yE), yF=cat(yF), yV=cat(yV), wE=cat(wE), wF=cat(wF), wV=cat(wV), nat=cat(nat),
         info=json.dumps(info))
print(json.dumps(info), file=sys.stderr)
