#!/usr/bin/env bash
# ace-jax's C++ FFI kernel in LAMMPS, 128 x 1, Si 4096 atoms: a candidate kernel source ($P/final.cc, here
# src/ace_jax/eval/ace_ffi.cc of branch perf/ffi-cpu-dag) and then the staged tree's kernel on the same node, per
# model (as run for "C++ kernel: SIMD lanes", docs/dev/accuracy-cost-results.md; devel's hour takes 3-4 models).
# $P/cpu-final.json: envs/sulis-cpu-lj-ffi.json with the handler at $P/<name>.so.  Rows: $P/retime-final.jsonl
# (-> results/sulis/sulis-cpu-128x1-ffi.jsonl, the fastest row per model) and $P/retime-plain.jsonl (-> ...-ffi-main).
#   sbatch accuracy_cost/envs/sulis-cpu-ffi-retime.sbatch ace/o4d20 ace/o4d16 ace/o4d14
#SBATCH --job-name=ffi-retime-cpu
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
ROOT=$HOME/bench-scaling; T=$ROOT/ace-jax-gracert; P=$ROOT/prof-ffi; PY=$ROOT/venv/bin/python
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0 OpenMPI/4.1.5
INC=$($PY -c 'import jaxlib, pathlib; print(pathlib.Path(jaxlib.__file__).parent / "include")')
CC="g++ -O3 -march=native -std=c++17 -shared -fPIC -isystem $INC"
B=$P/build-$SLURM_JOB_ID; mkdir -p $B
$CC $T/src/ace_jax/eval/ace_ffi.cc -o $B/libplain.so; $CC $P/final.cc -o $B/final.so
for v in plain final; do sed "s#prof-ffi/[a-z0-9]*\.so#prof-ffi/build-$SLURM_JOB_ID/$([ $v = plain ] && echo libplain || echo final).so#" $P/cpu-final.json > $B/cpu-$v.json; done
grep -o "ace_jax_direct=[^\"]*" $B/cpu-*.json
cd "$T/bench"
export PYTHONPATH=$T/bench:$T/src:$ROOT/lammps-jax-ffi/python JAX_PLATFORMS=cpu
export ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16 ACEJAX_AC_RSS_CAP_GB=400
export BENCH_LAMMPS_JAX_CPU=1 ACEJAX_AC_CPU_RANKS=128 ACEJAX_AC_CPU_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) final.cc $(md5sum < $P/final.cc)"
for c in "$@"; do
  st=1e-10; [ "$c" = ace/o3d8 ] && st=
  export ACEJAX_AC_ACE_SPLINE_TOL=$st
  for v in final plain; do
    echo "== $c $v $(date -Is)"
    $PY -m accuracy_cost.timing cpu --only $c --codes acejax-ace --modes lammps --redo \
        --env $B/cpu-$v.json --results $P/retime-$v.jsonl 2>&1 | tail -1 || true
  done
done
echo "done $(date -Is)"
