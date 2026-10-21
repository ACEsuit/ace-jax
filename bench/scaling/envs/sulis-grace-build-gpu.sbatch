#!/usr/bin/env bash
# LAMMPS (grace branch) for GRACE timing: sulis-grace.sh gpu.  Submit from bench/scaling/envs:
#   sbatch sulis-grace-build-gpu.sbatch
#SBATCH --job-name=grace-build-gpu
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
JOBS=$SLURM_CPUS_PER_TASK bash "$SLURM_SUBMIT_DIR/sulis-grace.sh" gpu      # $0 is the spooled copy
