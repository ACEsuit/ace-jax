#!/usr/bin/env bash
# CUDA (A100) LAMMPS with ML-YE3T (`pair_style ye3t/kk`) and stock ML-PACE (sulis-ye3t.sh gpu; after the CPU build).
# Submit from bench/scaling/envs:   sbatch sulis-ye3t-build-gpu.sbatch
#SBATCH --job-name=ye3t-build-gpu
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
JOBS=$SLURM_CPUS_PER_TASK bash "$SLURM_SUBMIT_DIR/sulis-ye3t.sh" gpu      # $0 is the spooled copy
