#!/usr/bin/env bash
# Per-kernel GPU time of ML-PACE pace/kk (product, as the benchmark runs it) on 4096 rattled diamond Si atoms,
# under Nsight Systems: one LAMMPS run per model.  Submit from ~/bench-scaling/ace-jax-ac/bench:
#   sbatch accuracy_cost/ml-pace/sulis-profile-gpu.sbatch NAME=PATH.yace ...
#SBATCH --job-name=pace-prof-gpu
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=$HOME/bench-scaling
W=$ROOT/ace-jax-ac/bench/accuracy_cost/ml-pace/prof-$SLURM_JOB_ID; mkdir -p "$W"; cd "$W"
LMP=${LMP:-$ROOT/lmp-pacetab.sh}
NSYS=/sulis/easybuild/software/CUDA/12.8.0/bin/nsys
for arg in "$@"; do
  name=${arg%%=*}; model=${arg#*=}
  cat > in.$name <<IN
units metal
boundary p p p
lattice diamond 5.43
region box block 0 8 0 8 0 8
create_box 1 box
create_atoms 1 box
mass 1 28.0855
displace_atoms all random 0.05 0.05 0.05 1234
pair_style pace product
pair_coeff * * $model Si
velocity all create 300 4928
timestep 0.001
fix 1 all nve
thermo 50
run 20
run 100
IN
  $NSYS profile -t cuda --stats=false --force-overwrite true -o prof_$name \
    $LMP -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.$name -log log.$name > out.$name 2>&1
  echo "== $name $(grep -A1 'Loop time' log.$name | tail -2 | head -1)"
  $NSYS stats -q --report cuda_gpu_kern_sum --format csv -o kern_$name prof_$name.nsys-rep > /dev/null
done
ls "$W"
