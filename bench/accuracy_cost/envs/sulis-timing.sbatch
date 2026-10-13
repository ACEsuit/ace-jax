#!/usr/bin/env bash
# Accuracy-cost timings on a Sulis A100 (bench/scaling's Sulis toolchain, envs/sulis.sh):
# the fitted models copied to $ACEJAX_AC_ROOT/runs/accuracy_cost, the code to
# ~/bench-scaling/ace-jax-ac (PYTHONPATH, not installed: the venv's editable ace-jax
# stays bench/scaling's).  No ML-QUIP in that LAMMPS: no GAP rows here.
#
#   sbatch -p gpu-devel -t 00:50:00 envs/sulis-timing.sbatch gpu ace/o3d8 --force-busy   # smoke
#   sbatch --exclusive envs/sulis-timing.sbatch gpu ace                                  # one family
#
# Rows: results/sulis-<device>.jsonl.  --exclusive for real timings: wait_idle reads the
# node's load1, which other jobs on a shared node raise.
#SBATCH --job-name=ac-timing
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
DEV=$1; ONLY=${2:-}; shift $(( $# < 2 ? $# : 2 ))
cd "$ROOT/ace-jax-ac/bench"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/ace-jax-ac/bench:${ACEJAX_SRC:-$ROOT/ace-jax-ac/src}   # ACEJAX_SRC: another ace-jax (e.g. ace-jax-sb, SBessel radials)
export ACEJAX_AC_ROOT=$ROOT/ac
# --exclusive on gpu-devel can expose both of a node's A100s; ACEJAX_AC_GPU=0 keeps one
[ -n "${ACEJAX_AC_GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$ACEJAX_AC_GPU
export ACEJAX_AC_MAX_LOAD1=16       # an idle exclusive 128-core node sits at load1 ~6
echo "job $SLURM_JOB_ID $DEV ${ONLY:-all} on $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
"$ROOT/venv/bin/python" -c "import ace_jax; print(ace_jax.__file__)"
"$ROOT/venv/bin/python" -m accuracy_cost.timing "$DEV" ${ONLY:+--only "$ONLY"} \
    --env accuracy_cost/envs/sulis-$DEV.json --results accuracy_cost/results/sulis-$DEV.jsonl "$@"
