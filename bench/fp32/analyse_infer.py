"""Plan step 1.4/1.5: fp32 (and TF32) ACECalculator results against fp64 (infer_dump.py outputs)."""
import json
import sys
from pathlib import Path

import numpy as np

DUMP = Path("/storage/eng/essswb/projects/ace-jax/runs/fp32_step1")


def load(m, tag):
    z = np.load(DUMP / f"infer_{m}_{tag}.npz")
    return {k: z[k] for k in z.files if k != "info"} | {"info": json.loads(str(z["info"]))}


res = []
for m in sys.argv[1:]:
    r64 = load(m, "f64")
    out = dict(model=m, t_step={t: load(m, t)["info"]["t_step_median"] for t in ("f64", "f32", "tf32")},
               model_s={t: load(m, t)["info"]["model_s"] for t in ("f64", "f32", "tf32")})
    for t in ("f32", "tf32"):
        r = load(m, t)
        dE = (r["E"] - r64["E"]) / r["nat"]
        dF = r["F"] - r64["F"]
        out[t] = dict(E_meV_atom_rms=1e3 * float(np.sqrt(np.mean(dE ** 2))), E_meV_atom_max=1e3 * float(np.abs(dE).max()),
                      F_meV_A_rms=1e3 * float(np.sqrt(np.mean(dF ** 2))), F_meV_A_max=1e3 * float(np.abs(dF).max()),
                      F_rel=float(np.linalg.norm(dF) / np.linalg.norm(r64["F"])),
                      S_rel=float(np.linalg.norm(r["S"] - r64["S"]) / np.linalg.norm(r64["S"])),
                      Ebig_meV_atom=1e3 * float(np.abs(r["Ebig"] - r64["Ebig"]).max() / len(r["Fbig"])),
                      Fbig_meV_A_max=1e3 * float(np.abs(r["Fbig"] - r64["Fbig"]).max()))
    res.append(out)
print(json.dumps(res, indent=1))
