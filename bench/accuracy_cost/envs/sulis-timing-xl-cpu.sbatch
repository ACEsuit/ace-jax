#!/usr/bin/env bash
# Symmetrix-XL MACE ("mace-xl", "mace-xl-f32") on a Sulis compute node, the CPU LAMMPS protocol of every code:
# RANKS MPI ranks x THREADS cores, bound (run_lammps.cpu_mpirun), `-k on t THREADS -sf kk`, the proc_grid
# `processors` grid.  Needs bench/scaling/envs/sulis-xl-build-cpu.sbatch (lmp-xl-cpu.sh, model-xl.json and its
# host R1 artifacts).  Args: RANKS THREADS [ONLY] [timing args...]; --codes adds the original "mace" for a
# like-for-like baseline.
#   sbatch accuracy_cost/envs/sulis-timing-xl-cpu.sbatch 128 1 mace     # rows: results/sulis-cpu-128x1-mace_xl.jsonl
#SBATCH --job-name=ac-xl-cpu
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
R=$1; T=$2; ONLY=${3:-mace}; shift $(( $# < 3 ? $# : 3 ))
cd "$ROOT/ace-jax-ac/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
export PYTHONPATH=$ROOT/ace-jax-ac/bench:$ROOT/ace-jax-ac/src
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 ACEJAX_AC_RSS_CAP_GB=400 CUDA_VISIBLE_DEVICES=
export ACEJAX_AC_CPU_RANKS=$R ACEJAX_AC_CPU_THREADS=$T
echo "job $SLURM_JOB_ID cpu ${R}x${T} ${ONLY} on $(hostname) $(date -Is)"; lscpu | grep "Model name"; cat $ROOT/VERSIONS-xl
"$ROOT/venv/bin/python" -m accuracy_cost.timing cpu --only "$ONLY" --codes mace-xl mace-xl-f32 \
    --env accuracy_cost/envs/sulis-cpu-xl.json --results accuracy_cost/results/sulis-cpu-${R}x${T}-mace_xl.jsonl "$@"
echo "done $(date -Is)"
