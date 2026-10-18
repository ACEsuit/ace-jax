#!/bin/bash
# Calibrate the RMSE-based row check on the four fits whose float32-vs-float64 RMSE is known.
D=/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost
E1X=docs/user/tutorials/data/school/e1x/reference.xyz
O=/storage/eng/essswb/projects/ace-jax/runs/fp32_step3/check_calib
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p $O
gap() { uv run --no-sync aj fit --train $D/train.xyz --order 4 --max-degree $2 --rcut 5.0 --virial-key virial \
  --weights $D/weights_acejax.json --e0 lsq --rungs map --m-per-species 0 --configs-per-batch 2 \
  --row-precision float32 --row-check --out $O/$1 > $O/$1.log 2>&1; echo "$1 $?"; }
e1x() { uv run --no-sync aj fit --train $E1X --order $2 --max-degree $3 --rcut 5.0 --virial-key virial --rungs map \
  --m-per-species 0 --row-precision float32 --row-check --out $O/$1 > $O/$1.log 2>&1; echo "$1 $?"; }
gap gap_o4d16 16; gap gap_o4d12 12; e1x e1x_o3d12 3 12; e1x e1x_o4d14 4 14
