#!/bin/bash
# Step-2 open issues: bundle step time vs block_rows over models / dtypes (A4500, matrix layout).
R=/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost
export BUNDLE_STEPS=${BUNDLE_STEPS:-100} BUNDLE_TMP=/storage/eng/essswb/projects/ace-jax/runs/fp32_step2/diag
run() { echo "# $*"; uv run --no-sync python bench/fp32/diag_blocks.py "$@" 2>&1 >/dev/null | grep '^{'; }
for dt in float32 float64; do
  run $R/ace_per_quantity_noise/o4d16/model.npz 32768 $dt 1024 2048 4096 0
  run $R/ace_per_quantity_noise/o3d8/model.npz 32768 $dt 1024 2048 4096 0
  run $R/pace/f1352/model.yace 32768 $dt 1024 2048 4096 0
  run $R/pace/f400/model.yace 32768 $dt 1024 2048 4096 0
done
run $R/ace_per_quantity_noise/o4d16/model.npz 262144 float32 1024 2048
run $R/pace/f400/model.yace 262144 float32 1024 2048
DIAG_LAYOUT=dense run $R/ace_per_quantity_noise/o4d16/model.npz 32768 float32 1024 2048 0
