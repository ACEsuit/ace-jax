#!/usr/bin/env bash
# Symmetrix-XL (MACE; github.com/bonan-group/symmetrix-xl, arXiv 2610.01036) on Sulis, the accuracy-cost
# benchmark's "mace-xl" code.  Its own LAMMPS trees (the release branch: XL needs >= 10 Dec 2025), never
# ~/bench-scaling/lammps*: the original-Symmetrix "mace" rows keep envs/sulis.sh's tree.
#
#   bash bench/scaling/envs/sulis-xl.sh sources venv      # login node: clone / download only
#   sbatch bench/scaling/envs/sulis-xl-build-cpu.sbatch    # py_cpu lammps_cpu wrappers models_cpu (compute node)
#   sbatch bench/scaling/envs/sulis-xl-build-gpu.sbatch    # py_cuda lammps_cuda wrappers models_cuda (an A100)
#
# Layout ($XL = ~/bench-scaling/symmetrix-xl):
#   src           symmetrix-xl $XL_REV, recursive (Kokkos 5.0.2 + KokkosKernels for its Python backend)
#   lammps-cpu    LAMMPS $LAMMPS_REV + pair_symmetrix (install.sh links it in), built by tools/symmetrix_build.py:
#   lammps-cuda     two copies, so the two builds never edit one tree at once
#   install-cpu   Kokkos OpenMP (Kokkos_ARCH_NATIVE: build on a compute node, EPYC 7742) + MPI
#   install-cuda  Kokkos CUDA sm_80 + Serial (the qualified XL GPU configuration) + MPI
#   venv-cpu      the XL Python frontend + CPU backend, mace-torch 0.3.16 (symmetrix_extract_mace) and the host
#                 JIT preparation (symmetrix_prepare_jit_host_artifact)
#   venv-cuda     frontend + CUDA sm_80 backend (symmetrix_prepare_jit_device_artifact)
#   jit-cache     SYMMETRIX_JIT_CACHE: content-addressed R1 artifacts, host (.so) and device (.cubin)
#   $ROOT/lmp-xl-cpu.sh, $ROOT/lmp-xl.sh   wrappers (modules + exec), the env jsons' `lmp_xl`
#
# Models (models_cpu): each MACE run dir's model-cpu.model -> model-xl.json (compact format v2,
# symmetrix_extract_mace's default), then R1 artifacts per precision recorded next to it as
# model-xl.jit-cpu.json / model-xl.jit-cuda.json ({precision: {"args": pair_style fragment} | {"error": why}}),
# which run_lammps.xl_pair_args reads.  A model the JIT refuses (R1 codegen takes only "uvu, [channels,1]"
# paths) keeps its error there and its rows are `unsupported`; nothing falls back to the non-compiled path.
#
# CMake: XL needs >= 3.27, Sulis modules stop at 3.26, so the venvs carry PyPI's cmake.  BLAS: OpenBLAS/0.3.23
# is pthreaded, which XL admits with one OpenMP worker -- the benchmark's 128 ranks x 1 thread.
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
XL=$ROOT/symmetrix-xl
XL_REV=aee9f980eb48bbee125a500a28fa6517466a1b66          # main, 11 Oct 2026 (symmetrix-xl 0.1.1+)
LAMMPS_REV=0165101ce505734e8fb249694b22e295a7ab4182      # release branch, "30 Sep 2026"
MACE_TORCH=0.3.16                                         # as the fits (bench/scaling's venv)
MODS="GCC/12.3.0 OpenMPI/4.1.5 OpenBLAS/0.3.23"         # foss/2023a parts, as envs/sulis.sh
CUDA_MOD=CUDA/12.6.0                                     # A100 nodes: driver 560.35 (CUDA 12.6); NVRTC artifacts must load there
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-16}}
MACE_DIR=${MACE_DIR:-$ROOT/ac/runs/accuracy_cost/mace}
export SYMMETRIX_JIT_CACHE=$XL/jit-cache UV_CACHE_DIR=$XL/uv-cache
export PATH=$HOME/.local/bin:$PATH
mkdir -p "$XL" "$SYMMETRIX_JIT_CACHE"
chmod -R go-w "$SYMMETRIX_JIT_CACHE"; chmod go-w "$ROOT" "$XL"   # XL refuses a cache, or an ancestor, others can write (umask 002)
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge; module load $MODS; set -u

pin() {                   # pin <dir> <url> <sha>: a checkout of exactly that commit
  local dir=$1 url=$2 sha=$3
  if [ ! -d "$dir/.git" ]; then
    git init -q "$dir"; git -C "$dir" remote add origin "$url"
  fi
  if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" != "$sha" ]; then
    git -C "$dir" fetch -q --depth 1 origin "$sha"
    git -C "$dir" checkout -q --detach FETCH_HEAD
  fi
}

sources() {
  [ -d "$XL/src/.git" ] || git clone -q https://github.com/bonan-group/symmetrix-xl.git "$XL/src"
  git -C "$XL/src" checkout -q "$XL_REV"; git -C "$XL/src" submodule update --init --recursive -q
  pin "$XL/lammps-cpu" https://github.com/lammps/lammps.git $LAMMPS_REV
  pin "$XL/lammps-cuda" https://github.com/lammps/lammps.git $LAMMPS_REV
}

venv() {                  # download only; the backends compile in py_cpu / py_cuda
  for v in venv-cpu venv-cuda; do [ -x "$XL/$v/bin/python" ] || uv venv -q -p 3.12 "$XL/$v"; done
  uv pip install -q -p "$XL/venv-cpu/bin/python" "cmake>=3.27,<4" ninja numpy scipy ase "mace-torch==$MACE_TORCH"
  uv pip install -q -p "$XL/venv-cuda/bin/python" "cmake>=3.27,<4" ninja numpy scipy ase
}

build() {                 # build <venv> <symmetrix_build.py args...>: the helper installs into the Python that runs it
  local v=$1; shift
  # PyPI cmake's bin/cmake is a Python shim; the helper's provenance check wants the binary it actually runs
  ( source "$XL/$v/bin/activate"; export PATH=$(python -c 'import cmake; print(cmake.CMAKE_BIN_DIR)'):$PATH
    cd "$XL/src"; python tools/symmetrix_build.py "$@" )
}

py_cpu() { build venv-cpu install --backend cpu --cpu-target native --build-root "$XL/build-py-cpu" --jobs "$JOBS"; }
lammps_cpu() {
  build venv-cpu lammps --backend cpu --cpu-target native --mpi on --source "$XL/lammps-cpu" \
        --prefix "$XL/install-cpu" --build-root "$XL/build-lmp-cpu" --jobs "$JOBS"
}
py_cuda() {               # a GPU backend installs over the frontend + CPU backend of the same source
  "$XL/venv-cuda/bin/python" -c "import symmetrix" 2>/dev/null ||
    build venv-cuda install --backend cpu --cpu-target native --build-root "$XL/build-py-cuda-host" --jobs "$JOBS"
  set +u; module load $CUDA_MOD; set -u
  build venv-cuda install --backend cuda --arch sm80 --cuda-root "$EBROOTCUDA" --build-root "$XL/build-py-cuda" --jobs "$JOBS"
  # the wheel takes any 12.x pip CUDA: 12.9's libcublas then binds the module's 12.6 libcublasLt (undefined
  # cublasLtGetEnvironmentMode) and 12.9 NVRTC emits PTX the 12.6 driver rejects; pin the pip libraries to 12.6
  uv pip install -q -p "$XL/venv-cuda/bin/python" "nvidia-cublas-cu12>=12.6,<12.7" "nvidia-cuda-nvrtc-cu12>=12.6,<12.7" \
      "nvidia-cuda-runtime-cu12>=12.6,<12.7"
  "$XL/venv-cuda/bin/python" -c "import symmetrix_backend_cuda12_sm80"
}
lammps_cuda() {
  set +u; module load $CUDA_MOD; set -u
  build venv-cuda lammps --backend cuda --arch sm80 --cuda-root "$EBROOTCUDA" --mpi on --source "$XL/lammps-cuda" \
        --prefix "$XL/install-cuda" --build-root "$XL/build-lmp-cuda" --jobs "$JOBS"
}

wrappers() {              # lmp-xl-cpu.sh / lmp-xl.sh: whichever installs exist
  local tag mods
  for tag in cpu cuda; do
    [ -x "$XL/install-$tag/bin/lmp" ] || continue
    mods=$MODS; [ $tag = cuda ] && mods="$MODS $CUDA_MOD"
    local out=$ROOT/lmp-xl-cpu.sh; [ $tag = cuda ] && out=$ROOT/lmp-xl.sh
    cat > "$out" <<EOF
#!/usr/bin/env bash
# Symmetrix-XL LAMMPS ($tag; bench/scaling/envs/sulis-xl.sh)
set +u; source /etc/profile.d/modules.sh 2>/dev/null; module purge >/dev/null 2>&1; module load $mods >/dev/null 2>&1; set -u
export SYMMETRIX_JIT_CACHE=$SYMMETRIX_JIT_CACHE
exec $XL/install-$tag/bin/lmp "\$@"
EOF
    chmod +x "$out"; "$out" -h | grep -i symmetrix | head -5
  done
}

jit() {                   # jit <cpu|cuda>: extract (cpu) and prepare R1 artifacts for every MACE run dir
  local tag=$1 v=venv-cpu; [ "$tag" = cuda ] && v=venv-cuda
  [ "$tag" = cuda ] && { set +u; module load $CUDA_MOD; set -u; }
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} "$XL/$v/bin/python" - "$tag" "$MACE_DIR" <<'PY'
import json, pathlib, subprocess, sys
tag, root = sys.argv[1], pathlib.Path(sys.argv[2])
bindir = pathlib.Path(sys.executable).parent
for d in sorted(p for p in root.iterdir() if (p / "model-cpu.model").exists()):
    src, out = d / "model-cpu.model", d / "model-xl.json"
    if tag == "cpu" and (not out.exists() or src.stat().st_mtime > out.stat().st_mtime):
        p = subprocess.run([bindir / "symmetrix_extract_mace", "--model", src, "--output", out],
                           capture_output=True, text=True)
        if p.returncode:
            print(d.name, "extract failed:", (p.stderr or p.stdout)[-800:], flush=True)
            out.unlink(missing_ok=True); continue
    if not out.exists():
        print(d.name, "no model-xl.json", flush=True); continue
    rec = {}
    for prec in ("float64", "float32"):
        if tag == "cpu":                              # host .so: deploy CPU = this compute node's (native)
            cmd = [bindir / "symmetrix_prepare_jit_host_artifact", "--model", out, "--precision", prec, "--path-only"]
        else:
            cmd = [bindir / "symmetrix_prepare_jit_device_artifact", "--model", out, "--precision", prec,
                   "--lammps-arguments"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        txt = p.stdout.strip()
        if p.returncode or not txt:
            rec[prec] = {"error": ((p.stderr or "") + txt)[-1500:]}
        else:
            rec[prec] = {"args": f"jit_host_artifact {txt.splitlines()[-1]}" if tag == "cpu" else " ".join(txt.split())}
        print(d.name, tag, prec, "ok" if "args" in rec[prec] else "FAILED " + rec[prec]["error"][-300:], flush=True)
    (d / f"model-xl.jit-{tag}.json").write_text(json.dumps(rec, indent=1) + "\n")
PY
}
models_cpu() { jit cpu; }
models_cuda() { jit cuda; }

versions() {
  {
    echo "symmetrix-xl $(git -C "$XL/src" log -1 --format='%h %cd')"
    git -C "$XL/src" submodule status | awk '{print "  " $2, substr($1,1,7), $3}'
    echo "lammps $(git -C "$XL/lammps-cpu" log -1 --format='%h %cd') ($(grep -o '"[^"]*"' "$XL/lammps-cpu/src/version.h" | head -1))"
    echo "modules $MODS (+ $CUDA_MOD for cuda); cmake $("$XL/venv-cpu/bin/cmake" --version | head -1)"
    echo "mace-torch $("$XL/venv-cpu/bin/python" -c 'import mace; print(mace.__version__)' 2>/dev/null)"
    for tag in cpu cuda; do
      [ -x "$XL/install-$tag/bin/lmp" ] && echo "install-$tag $(ls -l --time-style=+%F "$XL/install-$tag/bin/lmp" | awk '{print $6}')"
    done
  } > "$ROOT/VERSIONS-xl"
  cat "$ROOT/VERSIONS-xl"
}

steps=("$@"); [ ${#steps[@]} -gt 0 ] || steps=(sources venv)
for s in "${steps[@]}"; do echo "=== $s $(date -Is)"; "$s"; done
