#!/usr/bin/env bash
# GPUMD's nep trainer (sm_80) for the accuracy-cost NEP fits (sulis-nep.sh gpumd).
# Submit from bench/scaling/envs:  sbatch sulis-nep-build-gpu.sbatch
#SBATCH --job-name=nep-build-gpu
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
JOBS=$SLURM_CPUS_PER_TASK bash "$SLURM_SUBMIT_DIR/sulis-nep.sh" gpumd      # $0 is the spooled copy
