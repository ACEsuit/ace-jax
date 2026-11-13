#!/usr/bin/env bash
# CPU LAMMPS with ML-YE3T (pair_style ye3t) and stock ML-PACE (sulis-ye3t.sh cpu).
# Submit from bench/scaling/envs:  sbatch sulis-ye3t-build-cpu.sbatch
#SBATCH --job-name=ye3t-build-cpu
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
JOBS=$SLURM_CPUS_PER_TASK bash "$SLURM_SUBMIT_DIR/sulis-ye3t.sh" cpu      # $0 is the spooled copy
