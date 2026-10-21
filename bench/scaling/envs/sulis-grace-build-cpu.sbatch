#!/usr/bin/env bash
# LAMMPS (grace branch) for GRACE timing: sulis-grace.sh cpu.  Submit from bench/scaling/envs:
#   sbatch sulis-grace-build-cpu.sbatch
#SBATCH --job-name=grace-build-cpu
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
JOBS=$SLURM_CPUS_PER_TASK bash "$SLURM_SUBMIT_DIR/sulis-grace.sh" cpu      # $0 is the spooled copy
