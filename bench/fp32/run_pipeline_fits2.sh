#!/bin/bash
# Step 3 item 3, second round: float32 rows with float32 predictions; the opt-in check, revised.
D=/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost
O=/storage/eng/essswb/projects/ace-jax/runs/fp32_step3/pipeline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
run() {  # name order degree extra-args
  out=$O/$1
  /usr/bin/time -f "%e s wall, %M kB" uv run --no-sync aj fit --train $D/train.xyz --order $2 --max-degree $3 \
    --rcut 5.0 --virial-key virial --weights $D/weights_acejax.json --e0 lsq --rungs map --m-per-species 0 \
    --configs-per-batch 2 --out $out ${@:4} > $out.log 2>&1
  echo "$1 exit $? $(tail -1 $out.log)"
}
run o4d16_float32_v2 4 16 --row-precision float32
run o4d16_float32_check 4 16 --row-precision float32 --row-check
run o4d12_float32_check 4 12 --row-precision float32 --row-check
