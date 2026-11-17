#!/usr/bin/env bash
# Foundation MACE models (family mace_fm) on an A100 through Symmetrix v0.3.0's pair_style symmetrix/mace/foundation
# (float64), .../float32 and .../float32-mixed (bench/scaling/envs/sulis-symx3.sh lammps_cuda_fdn).  Each model is
# extracted once (model-fdn.json: the registered head, Si) and gated against the model in PyTorch (float64).
# Rows: results/sulis-gpu-fdn.jsonl.  Args: cases (family/size).  A shared node: --export=ALL,ACEJAX_AC_MAX_LOAD1=9999 (REDO=1 there re-gates and re-times).
#   sbatch accuracy_cost/envs/sulis-gpu-fdn.sbatch [mace_fm/mh1-omat ...]      (from ~/bench-scaling/ace-jax-fdn/bench)
#SBATCH --job-name=ac-gpu-fdn
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-fdn/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0
export PYTHONPATH=$ROOT/ace-jax-fdn/bench:$ROOT/ace-jax-fdn/src ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=${ACEJAX_AC_MAX_LOAD1:-16}
RESULTS=${RESULTS:-accuracy_cost/results/sulis-gpu-fdn.jsonl}
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; cat $ROOT/VERSIONS-symx3 || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
for c in ${@:-mace_fm}; do
  for s in $("$ROOT/venv/bin/python" -c "
import sys; from accuracy_cost import common
f, _, s = sys.argv[1].partition('/'); print(' '.join(e['size'] for e in common.load_sweep()[f] if not s or e['size'] == s))" $c); do
    PATH=$ROOT/venv-symx3/bin:$PATH "$ROOT/venv-symx3/bin/python" -m accuracy_cost.fit_mace_fm export $s || true
  done
  "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only $c --codes mace-fdn mace-fdn-f32 mace-fdn-f32m --modes lammps \
      --env accuracy_cost/envs/sulis-gpu-fdn.json --results "$RESULTS" ${REDO:+--redo} || true
done
echo "done $(date -Is)"
