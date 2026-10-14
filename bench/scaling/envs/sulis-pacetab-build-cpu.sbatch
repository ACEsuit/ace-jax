#!/usr/bin/env bash
# CPU LAMMPS with the tabulated-radial ML-PACE (sulis-pacetab.sh cpu).  Submit from bench/scaling/envs:
#   sbatch sulis-pacetab-build-cpu.sbatch [cpu|cpufast]          (default cpu)
#SBATCH --job-name=pacetab-build-cpu
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
JOBS=$SLURM_CPUS_PER_TASK bash "$SLURM_SUBMIT_DIR/sulis-pacetab.sh" ${1:-cpu}      # $0 is the spooled copy
