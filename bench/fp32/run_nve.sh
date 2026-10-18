#!/bin/bash
# NVE drift campaign on the local GPU (bench/fp32/nve_drift.py; docs/dev/fp32-results.md "When float32 is safe").
#   bench/fp32/run_nve.sh [stage ...]     stages: ladder dt liquid box d20 (default: all, in that order)
set -e
M=/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/ace
O=${OUT:-/storage/eng/essswb/projects/ace-jax/runs/fp32_step3/nve}
export BUNDLE_PYTHON=${BUNDLE_PYTHON:-/home/eng/essswb/bench-scaling/venv/bin/python} BUNDLE_TMP=$O
export PYTHONPATH=$(cd "$(dirname "$0")/.." && pwd):$(cd "$(dirname "$0")/../.." && pwd)/src
D="$BUNDLE_PYTHON bench/fp32/nve_drift.py"
cd "$(dirname "$0")/../.."
echo "start $(date -Is) $(git rev-parse --short HEAD)"
for s in ${*:-ladder dt liquid box d20}; do
  case $s in
    ladder) $D $M/o4d12/model.npz $M/o4d16/model.npz --ps 40 --out $O/ladder.jsonl ;;  # 4096 atoms, 1 fs
    dt)     $D $M/o4d12/model.npz --dts 0.5 2.0 --ps 20 --out $O/dt.jsonl ;;           # timestep scaling
    liquid) $D $M/o4d12/model.npz --temp 4000 --ps 20 --out $O/liquid.jsonl ;;         # melts: far from equilibrium
    box)    $D $M/o4d12/model.npz --sizes 32768 --ps 10 --out $O/box.jsonl ;;          # 87 A box: rij rounding
    d20)    $D $M/o4d20/model.npz --ps 10 --out $O/ladder.jsonl ;;                     # 0.1M atom-steps/s in f64 here
  esac
done
echo "done $(date -Is)"
