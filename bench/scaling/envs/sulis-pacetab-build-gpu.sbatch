#!/usr/bin/env bash
# CUDA (A100) LAMMPS with the tabulated-radial ML-PACE (sulis-pacetab.sh gpu).  Submit from bench/scaling/envs:
#   sbatch sulis-pacetab-build-gpu.sbatch [gpu|gputeam]          (default gpu)
#SBATCH --job-name=pacetab-build-gpu
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
JOBS=$SLURM_CPUS_PER_TASK bash "$SLURM_SUBMIT_DIR/sulis-pacetab.sh" ${1:-gpu}      # $0 is the spooled copy
