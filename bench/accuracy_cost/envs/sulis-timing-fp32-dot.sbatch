#!/usr/bin/env bash
# ace-jax float32 on a Sulis A100 with its matmuls at one jax DotAlgorithmPreset (bench/fp32/dot_algorithm.py;
# docs/dev/fp32-results.md): F32_F32_F32 is the shipped float32 (highest precision, no hook), TF32_TF32_F32_X3
# and TF32_TF32_F32 put the float32 contractions on the TF32 tensor cores.  Per size: precision_check against
# float64 (64 and 512 atoms) into results/sulis-gpu-fp32-dot-acc.jsonl, then the float32 acejax-ace LAMMPS row
# (spline_tol 1e-10, as the plotted rows) into results/sulis-gpu-fp32-dot-<preset>.jsonl.  The float64 parity
# gate is not rerun: its rows are copied from results/sulis-gpu-fp32-o4d*.jsonl (sulis-timing-fp32.sbatch).
#
#   sbatch envs/sulis-timing-fp32-dot.sbatch <preset> [size ...]     (from ace-jax-ac-fp32/bench/accuracy_cost)
#SBATCH --job-name=ac-fp32-dot
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu-devel
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
CODE=$ROOT/ace-jax-ac-fp32
PY=$ROOT/venv/bin/python
cd "$CODE/bench"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$CODE/bench:$CODE/src
export ACEJAX_AC_ROOT=$ROOT/ac
export ACEJAX_AC_MAX_LOAD1=16
export ACEJAX_AC_ACE_SPLINE_TOL=1e-10 ACEJAX_AC_ACE_DTYPES=float32
PRESET=$1; shift
SIZES=${*:-o4d12 o4d14 o4d16 o4d20}
OUT=accuracy_cost/results/sulis-gpu-fp32-dot-$PRESET.jsonl
ENV=accuracy_cost/envs/sulis-gpu-fp32.json
if [ "$PRESET" != F32_F32_F32 ]; then             # the hook in this process and in the driver's children
  export ACEJAX_EXP_DOT_ALGORITHM=$PRESET PYTHONPATH=$CODE/bench/fp32/dotalg:$PYTHONPATH
  ENV=accuracy_cost/results/sulis-gpu-fp32-dot-$PRESET.env.json
  "$PY" -c "import json,sys; e=json.load(open('accuracy_cost/envs/sulis-gpu-fp32.json')); \
e['pythonpath']='$CODE/bench/fp32/dotalg:'+e['pythonpath']; json.dump(e, open('$ENV', 'w'))"
fi
[ -e "$OUT" ] || cat accuracy_cost/results/sulis-gpu-fp32-o4d*.jsonl \
  | grep '"code": "acejax-ace"' | grep '"mode": "parity"' > "$OUT"
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) preset $PRESET sizes $SIZES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
for size in $SIZES; do
  "$PY" fp32/dot_algorithm.py "$ACEJAX_AC_ROOT/runs/accuracy_cost/ace/$size/model.npz" --cells 2 4 \
      | tee -a accuracy_cost/results/sulis-gpu-fp32-dot-acc.jsonl
  "$PY" -m accuracy_cost.timing gpu --only ace/$size --codes acejax-ace --env "$ENV" --results "$OUT"
done
