#!/usr/bin/env bash
# ace-jax's C++ kernel in LAMMPS, 128 x 1, Si 4096 atoms: ffi_precision="mixed" (the product basis in float32)
# and then float64, the same library and node, per model ("Mixed precision", docs/dev/accuracy-cost-results.md;
# devel's hour takes 3-4 models). The tree $T (src/ and bench/ of the branch) is staged by hand; $P/cpu-final.json
# is envs/sulis-cpu-lj-ffi.json with the handler at $P/<name>.so. Rows: $P/mixed-mixed.jsonl (-> results/sulis/
# sulis-cpu-128x1-ffi-mixed.jsonl) and $P/mixed-float64.jsonl (the same-node control).
#   sbatch accuracy_cost/envs/sulis-cpu-ffi-mixed.sbatch ace/o4d20 ace/o4d16 ace/o4d14
#SBATCH --job-name=ffi-mixed-cpu
#SBATCH --account=su007-jrk
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:59:00
#SBATCH --output=%x-%j.out
set -uo pipefail
ROOT=$HOME/bench-scaling; T=${T:-$ROOT/ace-jax-f32}; P=$ROOT/prof-ffi; PY=$ROOT/venv/bin/python
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
INC=$($PY -c 'import jaxlib, pathlib; print(pathlib.Path(jaxlib.__file__).parent / "include")')
B=$P/build-$SLURM_JOB_ID; mkdir -p $B
g++ -O3 -march=native -std=c++17 -shared -fPIC -Wno-psabi -isystem $INC $T/src/ace_jax/eval/ace_ffi.cc -o $B/lib.so
sed -e "s#prof-ffi/[a-z0-9]*\.so#prof-ffi/build-$SLURM_JOB_ID/lib.so#" -e "s#bench-scaling/ace-jax-[a-z0-9]*/#bench-scaling/$(basename $T)/#g" $P/cpu-final.json > $B/cpu.json
grep -o "ace_jax_direct=[^\"]*" $B/cpu.json
cd "$T/bench"
export PYTHONPATH=$T/bench:$T/src:$ROOT/lammps-jax-ffi/python JAX_PLATFORMS=cpu
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 ACEJAX_AC_RSS_CAP_GB=400
export BENCH_LAMMPS_JAX_CPU=1 ACEJAX_AC_CPU_RANKS=128 ACEJAX_AC_CPU_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) $(cat $T/REV 2>/dev/null)"
for c in "$@"; do
  st=1e-10; [ "$c" = ace/o3d8 ] && st=
  export ACEJAX_AC_ACE_SPLINE_TOL=$st
  for v in mixed float64; do
    echo "== $c $v $(date -Is)"
    ACEJAX_EXPORT_FFI_PRECISION=$([ $v = mixed ] && echo mixed) \
      $PY -m accuracy_cost.timing cpu --only $c --codes acejax-ace --modes lammps --redo \
        --env $B/cpu.json --results $P/mixed-$v.jsonl 2>&1 | tail -1 || true
  done
done
echo "done $(date -Is)"
