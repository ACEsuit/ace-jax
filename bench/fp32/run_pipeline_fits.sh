#!/bin/bash
# Step 3 item 3: aj fit end to end on GAP-18 Si, float64 vs float32 rows (A4500), as the accuracy-cost fits.
D=/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost
O=${OUT:-/storage/eng/essswb/projects/ace-jax/runs/fp32_step3/pipeline}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p $O
for b in ${BASES:-"4 12" "4 16"}; do
  set -- $b
  for rp in float64 float32; do
    out=$O/o$1d$2_$rp
    /usr/bin/time -f "%e s wall, %M kB" uv run --no-sync aj fit --train $D/train.xyz --order $1 --max-degree $2 \
      --rcut 5.0 --virial-key virial --weights $D/weights_acejax.json --e0 lsq --rungs map --m-per-species 0 \
      --configs-per-batch 2 --row-precision $rp --out $out > $out.log 2>&1
    echo "o$1d$2 $rp exit $? $(tail -1 $out.log)"
  done
done
