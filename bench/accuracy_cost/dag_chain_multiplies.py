"""results/dag_chain_multiplies.json: each ace-jax model's chain-form product-basis multiplies per site (and its DAG
plan's), the quantity aa_dag="auto" thresholds (`eval.aa_dag.AUTO_MIN_CHAIN_MUL`), from the lean model's aa_specs.
Runs where the fitted models are (/storage); ace_lr/o4d16 (fitted elsewhere) shares ace/o4d16's basis.

    PYTHONPATH=src python bench/accuracy_cost/dag_chain_multiplies.py
"""
import json
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
from ace_jax.eval.io import load
from ace_jax.eval.model import lean
from ace_jax.eval.aa_dag import build_plan, plan_stats
A = "/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/"
cases = [("ace", s, A + f"ace/{s}/model.npz") for s in ("o3d8", "o3d10", "o3d12", "o4d12", "o4d14", "o4d16", "o4d20")]
cases += [("ace_lr", s, A + f"ace_lr/{s}/model.npz") for s in ("o3d8", "o3d10", "o3d12", "o4d12", "o4d14", "o4d16")]
cases += [("ace_pshape", s, A + f"pace_shape/{s}_sb_fixed/model.npz") for s in ("md10", "md12", "md14", "md18")]
cases += [("ace_pshape_lr", s, A + f"pace_shape/{s}_sb_lr200/model.npz") for s in ("md10", "md12", "md14", "md18")]
out = []
import os
for fam, s, p in cases:
    if not os.path.exists(p): print("missing", p); continue
    m, meta, z = load(p)
    L = lean(m)
    row = {"family": fam, "size": s}
    for name in ("aa_specs", "blk_aa_specs"):
        specs = getattr(L, name, None)
        if specs is None: continue
        arrs = [np.asarray(a) for a in specs]
        n_A = 1 + max((int(a.max()) for a in arrs if a.size), default=0)
        st = plan_stats(arrs, build_plan(arrs, n_A, "greedy"))
        st.pop("per_level"); st["n_A"] = n_A; st["max_order"] = max(a.shape[1] for a in arrs if a.size)
        row[name] = st
    out.append(row); print(json.dumps(row), flush=True)
mul = {f"{r['family']}/{r['size']}": r["aa_specs"]["chain_mul"] for r in out}
mul.setdefault("ace_lr/o4d16", mul["ace/o4d16"])
dst = __import__("pathlib").Path(__file__).parent / "results" / "dag_chain_multiplies.json"
json.dump(dict(sorted(mul.items())), open(dst, "w"), indent=1)
