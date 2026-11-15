# Platform wheels with the compiled C++ kernel

ace-jax's wheels for Linux x86_64, Linux aarch64 and macOS arm64 carry the CPU kernel
(`eval/ace_ffi.cc`) compiled, so `ACECalculator(ffi=...)`, `BatchCalculator`, JAX-MD and CPU LAMMPS
bundles need no compiler. Other platforms take the pure wheel, which compiles the kernel on first use,
as every install did before. This note records the two measurements behind the design (2026-11-13,
main at 9115405) and how the pieces fit.

## Design

- `eval.ffi.build_prebuilt(dir)` compiles this machine's `VARIANTS` and writes `manifest.json`
  (source hash, jaxlib, machine, compiler, variants). `hatch_build.py` puts that directory in the wheel
  as `ace_jax/eval/_prebuilt/` and tags the wheel `py3-none-<platform>` (`ACEJAX_KERNEL_DIR`,
  `ACEJAX_KERNEL_PLAT`); the libraries are loaded with ctypes, so there is one wheel per platform, not
  per Python version. `kernel-wheels.yml` builds and tests them; `publish.yml` uploads them with the
  sdist and the pure wheel.
- `eval.ffi.prebuilt()` returns the library for this CPU, and a plain `library()` takes it. It is
  ignored, and the source compiled as before, when: flags are given (`flags=`, `ACEJAX_FFI_FLAGS`,
  `aj native --flags`, `aj export --ffi-flags`), `ACEJAX_FFI_PREBUILT=0`, the installed `ace_ffi.cc` is
  not the source the library was compiled from (the manifest's hash), the machine differs, or jaxlib is
  older than the one it was built against. So the Python that lays out the kernel's plan and the binary
  that reads it cannot drift apart.
- The GPU kernel stays source-only: a prebuilt one would need a CUDA 12/13 x architecture matrix, and
  the `cuda13` extra's nvcc wheel already removes the need for a toolkit.
- Linux libraries are built in manylinux_2_28 containers with `-static-libstdc++ -static-libgcc`; CI
  checks that they need only glibc <= 2.28. macOS libraries target 11.0.

## Which instruction set: x86-64-v3

Kernel time (`calc.last_timing["model_s"]`, median of 10 force calls after 4 warm-up calls, positions
moved inside the skin) for three Si fixtures, relative to `-O3 -march=native`; best of the rounds.
`bench/perf/kernel_flags.py`.

Xeon Silver 4216 (Cascade Lake, AVX-512; moriarty), GCC 11.5, 3 rounds:

| model | atoms | threads | native | x86-64-v3 | x86-64-v4 | x86-64-v2 | `-O3` |
|---|---|---|---|---|---|---|---|
| si_fitted (o3d10) | 4096 | 1 | 43.5 ms | 0.99x | 1.36x | 1.67x | 1.67x |
| si_fitted (o3d10) | 4096 | 16 | 14.5 ms | 1.01x | 1.10x | 1.27x | 1.25x |
| si_m710 (o4d14) | 1728 | 1 | 35.0 ms | 1.02x | 1.41x | 1.96x | 1.96x |
| si_m710 (o4d14) | 1728 | 16 | 9.2 ms | 1.03x | 1.14x | 1.43x | 1.39x |
| si_l2849 (o4d18) | 1000 | 1 | 52.5 ms | 1.03x | 1.36x | 1.90x | 1.89x |
| si_l2849 (o4d18) | 1000 | 16 | 10.0 ms | 1.05x | 1.20x | 1.51x | 1.49x |

EPYC 7742 (Zen 2; a Sulis compute node), GCC 12.3, pinned to 16 cores (`taskset -c 0-15`), 5 rounds:

| model | threads | native | x86-64-v3 | v3 `-mtune=znver2` | v3 `-mtune=generic` |
|---|---|---|---|---|---|
| si_fitted (o3d10) | 1 | 42.8 ms | 1.01x | 1.02x | 1.01x |
| si_fitted (o3d10) | 16 | 16.2 ms | 0.99x | 1.00x | 0.99x |
| si_m710 (o4d14) | 1 | 35.7 ms | 1.00x | 1.00x | 1.01x |
| si_m710 (o4d14) | 16 | 8.8 ms | 0.99x | 1.00x | 1.00x |
| si_l2849 (o4d18) | 1 | 52.5 ms | 1.02x | 1.00x | 1.02x |
| si_l2849 (o4d18) | 16 | 8.7 ms | 1.01x | 1.01x | 0.99x |

- **x86-64-v3 is native's speed**: within 5% on the Xeon and 2% on the EPYC. Tuning flags change
  nothing. That is the variant the wheels carry.
- **x86-64-v4 is 10-41% slower than native on the Xeon.** `-march=native` there is `cascadelake`,
  which prefers 256-bit vectors; generic AVX-512 code does not. No v4 variant.
- **The x86-64 baseline is 1.25-2x slower on the Xeon**, and 2-3.3x slower single-threaded on the
  EPYC (an earlier unpinned run, 2 rounds). The wheels carry it as the second variant, for CPUs
  without AVX2; it is still far faster than the XLA program.
- An unpinned 16-thread run on the 128-core EPYC node showed v3 at 1.13-1.22x native, with repeats of
  one build differing by up to 1.5x. Pinned, the difference is gone: it was thread placement.
- Apple M-series (clang, one round): `-O3` and `-O3 -march=native` agree within 2%. One variant.

## Which jaxlib: the oldest supported

The kernel is compiled against jaxlib's XLA FFI headers and registers with an API version. One library
built against each of jaxlib 0.10.1 (API 0.3) and 0.11.2 (API 0.4), loaded under both, on macOS arm64,
the Xeon and the EPYC:

| built against | under jaxlib 0.10.1 | under jaxlib 0.11.2 |
|---|---|---|
| 0.10.1 | runs | runs |
| 0.11.2 | refused: "the handler's API version (0.4) is incompatible with the framework's API version (0.3)" | runs |

Where it runs, energies, forces and stress agree with the JAX model to 1.7e-8 relative (the radial
table's tolerance). So the wheels' kernel is built against the oldest supported jaxlib
(`ffi.PREBUILT_JAXLIB`, which a test holds equal to pyproject's jax floor), and `prebuilt()` refuses an
older one. `kernel-wheels.yml` runs each wheel under that jaxlib and under the newest one (0.12.0.1
when this was written, checked locally). An XLA FFI major-version change would need a rebuild; the
installed package would then fail at registration, not silently.

Compiling one variant takes about 5 s and the library is about 190 KB.
