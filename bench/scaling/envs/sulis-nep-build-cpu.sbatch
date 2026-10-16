#!/usr/bin/env bash
# CPU LAMMPS with NEP_CPU's USER-NEP (pair_style nep) (sulis-nep.sh cpu).
# Submit from bench/scaling/envs:  sbatch sulis-nep-build-cpu.sbatch
#SBATCH --job-name=nep-build-cpu
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
JOBS=$SLURM_CPUS_PER_TASK bash "$SLURM_SUBMIT_DIR/sulis-nep.sh" cpu      # $0 is the spooled copy
