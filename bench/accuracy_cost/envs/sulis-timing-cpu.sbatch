#!/usr/bin/env bash
# Accuracy-cost timings on a Sulis node (GPU: an A100, sulis-timing.sbatch; CPU: this file's variant) (bench/scaling's Sulis toolchain, envs/sulis.sh):
# the fitted models copied to $ACEJAX_AC_ROOT/runs/accuracy_cost, the code to
# ~/bench-scaling/ace-jax-ac (PYTHONPATH, not installed: the venv's editable ace-jax
# stays bench/scaling's).  No ML-QUIP in that LAMMPS: no GAP rows here.
#
#   sbatch -p gpu-devel -t 00:50:00 envs/sulis-timing.sbatch gpu ace/o3d8 --force-busy   # smoke
#   sbatch --exclusive envs/sulis-timing.sbatch gpu ace                                  # one family
#
# Rows: results/sulis-<device>.jsonl.  --exclusive for real timings: wait_idle reads the
# node's load1, which other jobs on a shared node raise.
#SBATCH --job-name=ac-timing-cpu
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
DEV=$1; ONLY=${2:-}; shift $(( $# < 2 ? $# : 2 ))
cd "$ROOT/ace-jax-ac/bench"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
# CPU LAMMPS rows run 32 MPI ranks (timing.CPU_RANKS) through mpirun: OpenMPI on PATH, and 32 Slurm task slots
set +u; module load GCC/12.3.0 OpenMPI/4.1.5 >/dev/null 2>&1; set -u
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/ace-jax-ac/bench:${ACEJAX_SRC:-$ROOT/ace-jax-ac/src}   # ACEJAX_SRC: another ace-jax (e.g. ace-jax-sb, SBessel radials)
export ACEJAX_AC_ROOT=$ROOT/ac
export ACEJAX_AC_MAX_LOAD1=16       # an idle exclusive 128-core node sits at load1 ~6
echo "job $SLURM_JOB_ID $DEV ${ONLY:-all} on $(hostname) $(date -Is)"
lscpu | grep -E "Model name|^CPU\(s\)"; ~/bench-scaling/lmp.sh -h 2>&1 | head -3 || true   # the CUDA build on a GPU-less node
"$ROOT/venv/bin/python" -c "import ace_jax; print(ace_jax.__file__)"
"$ROOT/venv/bin/python" -m accuracy_cost.timing "$DEV" ${ONLY:+--only "$ONLY"} \
    --env accuracy_cost/envs/sulis-$DEV.json --results accuracy_cost/results/sulis-$DEV.jsonl "$@"
