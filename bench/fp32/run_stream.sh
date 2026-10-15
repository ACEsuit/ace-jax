#!/bin/bash
# Plan step 3 item 1: streamed QR factors of the full GAP-18 Si training set, three precisions.
R=/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/ace_per_quantity_noise
D=/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost
O=${OUT:-/storage/eng/essswb/projects/ace-jax/runs/fp32_step3}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p $O
for m in ${MODELS:-o3d12 o4d12 o4d16}; do
  for p in f64 f32rows f32; do
    uv run --no-sync python bench/fp32/stream_qr.py --prec $p --model $R/$m/model.npz --train $D/train.xyz \
      --weights $D/weights_acejax.json --out $O/${m}_$p.npz 2>&1 | grep -E '^\{|Error' | tail -2
  done
done
