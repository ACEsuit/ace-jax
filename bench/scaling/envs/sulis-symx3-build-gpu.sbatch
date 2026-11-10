#!/usr/bin/env bash
# Symmetrix v0.3.0 LAMMPS, Kokkos CUDA sm_80, on an A100 node.  After `bash sulis-symx3.sh sources venv`:
#   sbatch sulis-symx3-build-gpu.sbatch
#SBATCH --job-name=symx3-build-gpu
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
R=${BENCH_ROOT:-$HOME/bench-scaling}
for i in $(seq 120); do      # submitted while `sulis-symx3.sh sources venv` ran on the login node: wait for it
  [ -f "$R/VERSIONS-symx3" ] && true && break; sleep 30
done
[ -f "$R/VERSIONS-symx3" ] || { echo "sources not staged"; exit 1; }
JOBS=$SLURM_CPUS_PER_TASK bash "$SLURM_SUBMIT_DIR/sulis-symx3.sh" lammps_cuda
