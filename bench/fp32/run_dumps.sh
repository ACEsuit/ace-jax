#!/bin/bash
# Plan step 1: dump fp32 / fp32-TF32 / fp64 rows for the accuracy-cost Si bases (GPU).
set -e
R=/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/ace_per_quantity_noise
D=/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost
O=${OUT:-/storage/eng/essswb/projects/ace-jax/runs/fp32_step1}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p $O
for m in ${MODELS:-o3d8 o3d12 o4d12 o4d16}; do
  for v in "32 highest f32" "32 default tf32" "64 highest f64"; do
    set -- $v
    uv run --no-sync python bench/fp32/rows_dump.py --prec $1 --matmul $2 --model $R/$m/model.npz \
      --train $D/train.xyz --weights $D/weights_acejax.json --out $O/${m}_$3.npz 2>&1 | grep '^{'
  done
done
