"""Plan step 1.5: host fp64 QR merge time (stats._host_merge's dgeqrf) per buffered L rows and per
batch of rows, against the GPU row time per batch from the rows_dump outputs.  Run on an idle box."""
import json
import os
import sys

from analyse import load, merge_time

out = []
for m in sys.argv[1:]:
    r = load(m, "f64")
    L = r["E"].shape[1]
    rows = int(round(sum(len(r[k]) for k in "EFV") / r["info"]["n_batches"]))
    t_L = merge_time(L, L)
    out.append(dict(model=m, L=L, threads=os.environ.get("OPENBLAS_NUM_THREADS", "default"),
                    rows_per_batch=rows, t_merge_L_rows=t_L, t_merge_per_batch_amortised=t_L * rows / L,
                    t_gpu_rows_f32=load(m, "f32")["info"]["t_batch_median"],
                    t_gpu_rows_f64=r["info"]["t_batch_median"]))
print(json.dumps(out, indent=1))
