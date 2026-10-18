"""Plan step 1.3: the theta_MAP shift implied by lml_check's gradient errors, -dgrad / H_kk, with
H_kk the fp64 LML's diagonal curvature by second differences (h = 0.05)."""
import json
import sys

import numpy as np

from analyse import FIT, lml_terms, load

h = 0.05
lml = {r["model"]: r for r in json.load(open(sys.argv[1]))}
out = []
for m in sys.argv[2:]:
    r64, z = load(m, "f64"), np.load(FIT / m / "model.npz")
    th = json.load(open(FIT / m / "theta_map.json"))
    f0 = lml_terms(r64, th, z, np.float64)
    H = {k: (lml_terms(r64, th | {k: th[k] + h}, z, np.float64) - 2 * f0
             + lml_terms(r64, th | {k: th[k] - h}, z, np.float64)) / h ** 2
         for k in ("log_sigma_c", "log_sigma_E", "log_sigma_F", "log_sigma_V")}
    out.append(dict(model=m, H=H, shift={t: {k: -lml[m][t]["dgrad"][k] / H[k] for k in H}
                                         for t in ("f32rows_qr64", "f32rows_qr32")}))
print(json.dumps(out, indent=1))
