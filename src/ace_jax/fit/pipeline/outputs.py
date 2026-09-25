import csv
import dataclasses
import json
import pathlib

import numpy as np


def _dump(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)


def _write_stage(out, name, payload):
    if name == "data" and payload.perm is not None:
        np.save(out / "split_perm.npy", payload.perm)
    elif name == "map":
        _dump(out / "theta_map.json", payload.theta._asdict())
        if payload.restarts is not None:
            _dump(out / "map_restarts.json", payload.restarts)
        if payload.sigma_type_ratios is not None:
            _dump(out / "sigma_type_ratios.json", np.asarray(payload.sigma_type_ratios).tolist())
    elif name == "rungs":
        for rung, dr in payload.draws.items():
            np.save(out / f"draws_{rung}.npy", dr)
        if "laplace" in payload.info:
            _dump(out / "laplace_info.json", payload.info["laplace"])
        if "nuts" in payload.info:
            _dump(out / "nuts_summary.json", payload.info["nuts"])


def checkpoint_writer(out):
    """An on_stage callback for fit(): writes split_perm.npy, theta_map.json,
    map_restarts.json, draws_*.npy, laplace_info.json as each stage finishes (the
    run.py layout), so they survive a failure in a later stage."""
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    return lambda name, payload: _write_stage(out, name, payload)


def _metrics_csv(path, metrics, split):
    rows = [{"rung": key.split("/")[1], "quantity": q, **m}
            for key, per_q in metrics.items() if key.startswith(f"{split}/")
            for q, m in per_q.items()]
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def write_outputs(res, out, layout=("run",), argv=None):
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    cfg, d, b = res.config, res.data, res.built
    _dump(out / "theta_map.json", res.theta._asdict())
    for rung, dr in res.rungs.draws.items():
        np.save(out / f"draws_{rung}.npy", dr)
    if "nuts" in res.rungs.info:
        _dump(out / "nuts_summary.json", res.rungs.info["nuts"])
    if "run" in layout:
        if d.perm is not None:
            np.save(out / "split_perm.npy", d.perm)
        if res.map.restarts is not None:
            _dump(out / "map_restarts.json", res.map.restarts)
        if res.map.sigma_type_ratios is not None:
            _dump(out / "sigma_type_ratios.json", np.asarray(res.map.sigma_type_ratios).tolist())
        if "laplace" in res.rungs.info:
            _dump(out / "laplace_info.json", res.rungs.info["laplace"])
        for key, arr in res.preds.arrays.items():
            split, rung = key.split("/")
            np.savez(out / f"pred_{split}_{rung}.npz", **arr)
        if "ridge_scores" in res.preds.pops:
            _dump(out / "pops_ridge.json", res.preds.pops["ridge_scores"])
        if "envelope" in res.preds.pops:
            np.savez(out / "pops_envelope_test.npz", **res.preds.pops["envelope"])
        _dump(out / "metrics.json", res.preds.metrics)
        _dump(out / "timings.json", res.timings)
        els = [int(e) for e in d.meta["elements"]]
        _dump(out / "config.json", {**(argv or dataclasses.asdict(cfg)), "M": int(b.prob.ind.XM.shape[0]),
                                    "len_basis": b.gpcfg.len_basis,
                                    "E0": dict(zip(map(str, els), map(float, d.E0)))})
    if "cli" in layout:
        _metrics_csv(out / "metrics.csv", res.preds.metrics, "test")
        _metrics_csv(out / "metrics_ood.csv", res.preds.metrics, "ood")
        if "run" not in layout:
            _dump(out / "config.json", {**(argv or {}), "M": int(b.prob.ind.XM.shape[0]),
                                        "len_basis": b.gpcfg.len_basis, "n_train": len(d.train),
                                        "n_test": len(d.test)})
