"""Plan step 3, item 1: the streamed QR statistics of a whole training set, as the fit builds
them (`stats.device_qr_stream`, its batched merges), in one of three precisions:

    f64      rows and merges in float64 (the reference)
    f32rows  float32 rows (model and rij cast), float64 merges
    f32      everything float32 (x64 off)

The target rides along as an extra column of the rows, so each quantity's factor is the
(L+1) x (L+1) triangle [[R, c], [0, rho]]: rho is the residual norm carried through the merges,
where the library's QRStats keeps y'y separately and the LML forms y'y - v'v (a difference that
cancels when the fit nearly interpolates).  Saves the factors and the scalars to an npz.

    uv run --no-sync python bench/fp32/stream_qr.py --prec f32 --model M.npz --train T.xyz --out o.npz
"""
import argparse
import json
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--prec", choices=["f64", "f32rows", "f32"], required=True)
p.add_argument("--model", required=True)
p.add_argument("--train", required=True)
p.add_argument("--weights", default=None)
p.add_argument("--per-batch", type=int, default=2)
p.add_argument("--stride", type=int, default=1)
p.add_argument("--merge-rows", type=int, default=None, help="stats.merge_target override (0: every batch)")
p.add_argument("--out", required=True)
a = p.parse_args()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", a.prec != "f32")
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ace_jax.eval import load, with_dtype  # noqa: E402
from ace_jax.fit.data import build_dataset, load_configs  # noqa: E402
from ace_jax.fit.inducing import GPConfig  # noqa: E402
from ace_jax.fit.rows import linear_rows_bounded  # noqa: E402
from ace_jax.fit.stats import device_qr_stream  # noqa: E402

model, meta, z = load(a.model)
cfgs = load_configs(a.train, "energy", "forces", "virial", weights=a.weights and json.load(open(a.weights)))
cfgs = cfgs[::a.stride]
ds = build_dataset(cfgs, meta, np.asarray(z["E0"], np.float64), a.per_batch, log=lambda *_: None)
if a.prec == "f32rows":                                   # rows in float32; targets, weights, merges float64
    model = with_dtype(model, jnp.float32)
    ds = ds._replace(rij=ds.rij.astype(jnp.float32))
cfg = GPConfig(r0=2.4, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
               NZ=len(meta["elements"]), C=int(ds.y_E.shape[1]))
L = cfg.len_basis
fdt = jnp.float32 if a.prec == "f32" else jnp.float64


def rows_fn(batch):
    r = linear_rows_bounded(model, cfg, batch)
    parts, extra = [], []
    for P, y, w in ((r.E, batch.y_E, batch.w_E),
                    (r.F.reshape(-1, L), batch.y_F.reshape(-1), jnp.repeat(batch.w_F, 3)),
                    (r.V.reshape(-1, L), batch.y_V.reshape(-1), jnp.repeat(batch.w_V, 6))):
        yw, live = y * w, w > 0
        Pw = jnp.concatenate([P * w[:, None], yw[:, None]], 1).astype(fdt)   # y as column L
        parts.append((Pw, jnp.zeros(Pw.shape[0], fdt)))
        extra.append((yw @ yw, live.sum().astype(fdt),
                      jnp.sum(jnp.where(live, 2.0 * jnp.log(jnp.where(live, w, 1.0)), 0.0))))
    return parts, extra


init = [(np.zeros((L + 1, L + 1)), np.zeros(L + 1))] * 3
z0 = jnp.zeros((), fdt)
run = jax.jit(lambda ds: device_qr_stream(rows_fn, ds, init, [(z0, z0, z0)] * 3, a.merge_rows))
t = time.perf_counter()
out, ex = run(ds)
jax.block_until_ready(out)
dt = time.perf_counter() - t
save = {}
for q, (Rt, _), (yy, n, lw) in zip("EFV", out, ex):
    save[f"Rt_{q}"] = np.asarray(Rt, np.float64)
    save[f"yy_{q}"], save[f"n_{q}"], save[f"logw_{q}"] = (float(yy), float(n), float(lw))
info = dict(prec=a.prec, model=a.model, L=L, n_configs=len(cfgs), n_batches=int(ds.n_batches),
            batch_shape=list(ds.nbr.shape[1:]), seconds=dt, device=str(jax.devices()[0]),
            dtype=str(out[0][0].dtype), merge_rows=a.merge_rows)
np.savez(a.out, info=json.dumps(info), **save)
print(json.dumps(info), file=sys.stderr)
