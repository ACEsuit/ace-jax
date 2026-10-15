#!/usr/bin/env bash
# pace/kk on one A100, profiled (bench/mlpace-fast; build: scaling/envs/sulis-pacetab.sh gputeamsum), for each
# PACE_KK_RHOW in $RHOW_MODES (default "0 2 3": stock ComputeRho/Weights, flat offsets, flat offsets + the
# segmented backward; 1 is the fused team kernel):
#   0. plain timings (no profiler), Si o4d12-o4d20, 32,768 atoms
#   1. nsys per-kernel split, Si o4d12 / o4d20
#   2. ncu on the product-basis and per-neighbour kernels (Si o4d20, 4096 atoms, 1 step), if the node allows
#      GPU counters (desktop GPUs here do not: RmProfilingAdminOnly)
#   3. summed_neighbour_A vs the expanded .yace, a 3-element embedded model at order 3 and 4 (13,824 atoms)
# Inputs in $ROOT/pacekk-profile (in.si, in.time, cell_gpu.data, emb_{o3d8,o4d10}_{summed,expanded}.yace);
# outputs in $ROOT/pacekk-profile/out-$SLURM_JOB_ID.
#   sbatch accuracy_cost/envs/sulis-gpu-pacekk-profile.sbatch      (from ~/bench-scaling/ace-jax-ac/bench)
#SBATCH --job-name=pacekk-profile
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu-devel
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
P=$ROOT/pacekk-profile; O=$P/out-$SLURM_JOB_ID; mkdir -p $O; cd $O
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5 CUDA/12.8.0
LMP=$ROOT/lammps-pacekk-teamsum/build-kk/lmp
export LD_LIBRARY_PATH=$ROOT/lammps-pacekk-teamsum/build-kk:${LD_LIBRARY_PATH:-}
K="-k on g 1 -sf kk -pk kokkos newton on neigh half"
NSYS=$(command -v nsys || ls -d $EBROOTCUDA/bin/nsys $EBROOTCUDA/nsight-systems*/bin/nsys 2>/dev/null | head -1)
NCU=$(command -v ncu || ls -d $EBROOTCUDA/bin/ncu $EBROOTCUDA/nsight-compute*/ncu 2>/dev/null | head -1)
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,driver_version --format=csv
echo "nsys=${NSYS:-none} ncu=${NCU:-none} RmProfilingAdminOnly=$(grep -h RmProfilingAdminOnly /proc/driver/nvidia/params 2>/dev/null)"
M=$ROOT/ac/runs/accuracy_cost/ace_tab
loop() { grep "Loop time" $1 | tail -1 | awk '{printf "%.4f us/atom-step\n", 1e6*$4/($9*$12)}'; }
MODES=${RHOW_MODES:-0 2 3}

echo "== 0. timings (no profiler)"
for s in o4d12 o4d14 o4d16 o4d20; do for r in $MODES; do
  st=20; [ $s = o4d20 ] && st=10
  PACE_KK_RHOW=$r $LMP $K -in $P/in.si -var nc 16 -var steps $st -var warm 5 -var model $M/$s/model.yace -var pstyle product \
      -log tlog_si_${s}_r$r -echo none -screen none -nocite >/dev/null 2>&1
  echo "-- $s rhow$r $(loop tlog_si_${s}_r$r)"
done; done

echo "== 1. nsys kernel split"
for s in o4d12 o4d20; do for r in $MODES; do
  st=20; [ $s = o4d20 ] && st=10
  if [ -n "$NSYS" ]; then
    PACE_KK_RHOW=$r $NSYS profile -o nsys_${s}_r$r -t cuda --force-overwrite true $LMP $K -in $P/in.si -var nc 16 -var steps $st \
        -var warm 2 -var model $M/$s/model.yace -var pstyle product -log log_${s}_r$r -echo none -screen none -nocite >/dev/null 2>&1
    echo "-- $s rhow$r $(loop log_${s}_r$r)"
    $NSYS stats -r cuda_gpu_kern_sum -f csv nsys_${s}_r$r.nsys-rep 2>/dev/null | python3 -c "
import csv, re, sys
r = list(csv.DictReader(l for l in sys.stdin if l.count(',') > 3))
for x in r[:8]:
    m = re.search(r'TagPairPACE(\w+)', x['Name'])
    print(f\"   {float(x['Time (%)']):5.1f}%  {int(x['Total Time (ns)'])/1e6:9.1f} ms  {int(x['Instances']):5d}  {m.group(1) if m else x['Name'][:70]}\")"
  fi
done; done

echo "== 2. ncu (o4d20, 4096 atoms, 1 step)"
if [ -n "$NCU" ]; then
  for r in $MODES; do
    PACE_KK_RHOW=$r timeout 900 $NCU --target-processes all --kernel-name-base demangled \
        -k regex:"ComputeWeights|ComputeRho|RhoWeightsTeam|ComputeAi|ComputeDerivative" -c 4 \
        --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats \
        --section ComputeWorkloadAnalysis -o ncu_o4d20_r$r -f \
        $LMP $K -in $P/in.si -var nc 8 -var steps 1 -var warm 0 -var model $M/o4d20/model.yace -var pstyle product \
        -log /dev/null -echo none -screen none -nocite > ncu_o4d20_r$r.log 2>&1
    echo "-- rhow$r exit $?"; tail -3 ncu_o4d20_r$r.log
    [ -f ncu_o4d20_r$r.ncu-rep ] && $NCU --import ncu_o4d20_r$r.ncu-rep --page details --csv > ncu_o4d20_r$r.csv 2>/dev/null && wc -l ncu_o4d20_r$r.csv
  done
fi

echo "== 3. summed vs expanded (3-element embedded, 13,824 atoms)"
for m in o3d8 o4d10; do for y in summed expanded; do for r in $MODES; do
  PACE_KK_RHOW=$r $LMP $K -in $P/in.time -var cell $P/cell_gpu.data -var model $P/emb_${m}_$y.yace -var pstyle product \
      -var steps 20 -log tlog_${m}_${y}_r$r -echo none -screen none -nocite >/dev/null 2>&1
  echo "-- $m $y rhow$r $(loop tlog_${m}_${y}_r$r)"
done; done; done
echo "done $(date -Is)"
