#!/usr/bin/env bash
# CPU LAMMPS with Seko's POLYMLP package (pair_style polymlp) (sulis-polymlp.sh cpu).  The compiled-in invariant
# tables (polymlp_gtinv_data*.cpp) take the package's README "ten minutes to one hour" to compile.
# Submit from bench/scaling/envs:  sbatch sulis-polymlp-build-cpu.sbatch
#SBATCH --job-name=polymlp-build-cpu
#SBATCH --account=su007-jrk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ulimit -s unlimited || true
JOBS=$SLURM_CPUS_PER_TASK bash "$SLURM_SUBMIT_DIR/sulis-polymlp.sh" cpu      # $0 is the spooled copy
