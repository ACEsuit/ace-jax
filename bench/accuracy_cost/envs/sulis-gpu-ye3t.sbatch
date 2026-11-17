#!/usr/bin/env bash
# YE3T on an A100 through LAMMPS `pair_style ye3t/kk` (ye3t-lammps, FP64 Kokkos CUDA, device only; built by
# bench/scaling/envs/sulis-ye3t.sh gpu), with stock ML-PACE's pace/kk in the same binary.  Families as
# sulis-cpu-ye3t.sbatch: ace_cheb (codes ye3t-sym and mlpace on the same ChebExpCos .yace, gated against ace-jax on
# the source fit) and the fitted ye3t_* families (code ye3t, gated against ye3t-methods' ASE calculator on the CPU).
# Rows: results/sulis-gpu-ye3t.jsonl.  Args: cases (family/size).
#   sbatch accuracy_cost/envs/sulis-gpu-ye3t.sbatch [ace_cheb/o4d12 ...]      (from ~/bench-scaling/ace-jax-ye3t/bench)
#SBATCH --job-name=ac-gpu-ye3t
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
cd "$ROOT/ace-jax-ye3t/bench"
set +u; source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5; set -u   # as the CPU job: ye3t-methods' native evaluator (the gate's reference) needs this libstdc++
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/ace-jax-ye3t/bench:$ROOT/ace-jax-ye3t/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=${ACEJAX_AC_MAX_LOAD1:-16}       # a shared node: submit with a higher limit (rows record load1)
RESULTS=${RESULTS:-accuracy_cost/results/sulis-gpu-ye3t.jsonl}
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; cat $ROOT/VERSIONS-ye3t $ROOT/VERSIONS-ye3t-gpu || true
nvidia-smi --query-gpu=name,driver_version --format=csv || true
for c in ${@:-ace_cheb/o3d12 ace_cheb/o4d12 ace_cheb/o4d14 ace_cheb/o4d16 ace_cheb/o4d20 ye3t_sym_fit/r3-34 ye3t_sym_fit/r3-110 ye3t_sym_fit/r4-205 ye3t_sym_fit/r4-490 ye3t_sym_wfit/r3-34 ye3t_sym_wfit/r3-110 ye3t_sym_wfit/r4-205 ye3t_sym_wfit/r4-490 ye3t_mix_T/t8-284}; do
  "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only $c --codes ye3t-sym ye3t mlpace ${REDO:+--redo} \
      --env accuracy_cost/envs/sulis-gpu-ye3t.json --results "$RESULTS" || true
done
echo "done $(date -Is)"
