#!/usr/bin/env bash
# LAMMPS (grace-runtime branch) for the CPU timing: sulis-grace-runtime.sh cpu, on a compute node (-march=native).
# Submit from bench/scaling/envs after `bash sulis-grace-runtime.sh src` on the login node:
#   sbatch sulis-grace-runtime-build-cpu.sbatch
#SBATCH --job-name=gracert-build-cpu
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3850
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
JOBS=$SLURM_CPUS_PER_TASK bash "$SLURM_SUBMIT_DIR/sulis-grace-runtime.sh" cpu      # $0 is the spooled copy
