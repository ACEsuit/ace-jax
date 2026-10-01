# Third-party components bundled in ace-jax-coupling wheels

The wheel's `ace_jax_coupling/_lib/` holds a `juliac --trim` compiled library
(`libetcouple`) and the shared libraries it loads at runtime. The list below is
exactly the set traced as loaded by `coupling/tools/prune_bundle.py` (Julia
1.13.0, macOS arm64 build, traced with an empty HOME so artifacts resolve from the bundle; Linux builds load the same components). Full licence
texts are in `licenses/` next to this file. SPDX identifiers follow each
upstream's own licence file and Julia's `THIRDPARTY.md`
(`licenses/Julia-THIRDPARTY.md`).

**Versions and source.** Each third-party library comes from a Julia binary
package (JLL) at the exact build listed. Every JLL is built by a public recipe in
[JuliaPackaging/Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil) from the
upstream release of that version, which is the corresponding source, for example
for the LGPL libraries. The versions follow `coupling/julia/Manifest.toml` and
Julia 1.13.0's standard library; `tests/test_coupling_notices.py` checks them.

| Component | Version | Bundled file(s) | Licence (SPDX) | Text |
|---|---|---|---|---|
| ace-jax glue (Python + Julia shim) | ace-jax-coupling 0.2.0 | `ace_jax_coupling/*.py`, `libetcouple` | MIT | `LICENSE` |
| EquivariantTensors.jl (compiled into `libetcouple`; fork rev in `build_info()`) | 0.5.1 at jameskermode/EquivariantTensors.jl `1f24438` (exact commit: `build_info()`) | `libetcouple` | MIT | `licenses/EquivariantTensors.txt` |
| Julia runtime and standard library | Julia 1.13.0 | `libjulia`, `libjulia-internal` (privatized names), `libetcouple` | MIT (+ components listed in THIRDPARTY.md) | `licenses/Julia.txt`, `licenses/Julia-THIRDPARTY.md` |
| libunwind | 1.8.3 (LibUnwind_jll 1.8.3+1) | `libunwind.1` | MIT | `licenses/libunwind.txt` |
| OpenBLAS (incl. reference LAPACK) | 0.3.30 (OpenBLAS_jll 0.3.30+0) | `libopenblas64_` | BSD-3-Clause | `licenses/OpenBLAS.txt` |
| libblastrampoline | 5.15.0 (libblastrampoline_jll 5.15.0+0) | `libblastrampoline.5` | MIT | `licenses/libblastrampoline.txt` |
| openlibm | 0.8.7 (OpenLibm_jll 0.8.7+0) | `libopenlibm.4` | MIT AND BSD-2-Clause AND ISC | `licenses/openlibm.txt` |
| GMP | 6.3.0 (GMP_jll 6.3.0+2) | `libgmp.10` | LGPL-3.0-or-later OR GPL-2.0-or-later | `licenses/LGPL-3.0.txt`, `licenses/GPL-3.0.txt`, `licenses/GPL-2.0.txt` |
| MPFR | 4.2.2 (MPFR_jll 4.2.2+0) | `libmpfr.6` | LGPL-3.0-or-later | `licenses/LGPL-3.0.txt`, `licenses/GPL-3.0.txt` |
| PCRE2 | 10.46 (PCRE2_jll 10.46.0+0) | `libpcre2-8.0` | BSD-3-Clause WITH PCRE2-exception | `licenses/PCRE2.txt` |
| zlib | 1.3.1 (Zlib_jll 1.3.1+2) | `libz.1` | Zlib | `licenses/zlib.txt` |
| zstd | 1.5.7 (Zstd_jll 1.5.7+1) | `libzstd.1` | BSD-3-Clause (dual GPL-2.0; used under BSD) | `licenses/zstd.txt` |
| SuiteSparse_config | 7.10.1 (SuiteSparse_jll 7.10.1+0) | `libsuitesparseconfig.7` | BSD-3-Clause | `licenses/SuiteSparse_config.txt` |
| SuiteSparse AMD | 7.10.1 (SuiteSparse_jll 7.10.1+0) | `libamd.3` | BSD-3-Clause | `licenses/SuiteSparse-AMD.txt` |
| SuiteSparse CAMD | 7.10.1 (SuiteSparse_jll 7.10.1+0) | `libcamd.3` | BSD-3-Clause | `licenses/SuiteSparse-CAMD.txt` |
| SuiteSparse CCOLAMD | 7.10.1 (SuiteSparse_jll 7.10.1+0) | `libccolamd.3` | BSD-3-Clause | `licenses/SuiteSparse-CCOLAMD.txt` |
| SuiteSparse COLAMD | 7.10.1 (SuiteSparse_jll 7.10.1+0) | `libcolamd.3` | BSD-3-Clause | `licenses/SuiteSparse-COLAMD.txt` |
| SuiteSparse BTF | 7.10.1 (SuiteSparse_jll 7.10.1+0) | `libbtf.2` | LGPL-2.1-or-later | `licenses/SuiteSparse-BTF.txt`, `licenses/LGPL-2.1.txt` |
| SuiteSparse KLU | 7.10.1 (SuiteSparse_jll 7.10.1+0) | `libklu.2` | LGPL-2.1-or-later | `licenses/SuiteSparse-KLU.txt`, `licenses/LGPL-2.1.txt` |
| SuiteSparse LDL | 7.10.1 (SuiteSparse_jll 7.10.1+0) | `libldl.3` | LGPL-2.1-or-later | `licenses/SuiteSparse-LDL.txt`, `licenses/LGPL-2.1.txt` |
| SuiteSparse CHOLMOD, UMFPACK, SPQR, RBio: **not bundled** | placeholders (ace-jax-coupling 0.2.0) | `libcholmod.5`, `libumfpack.6`, `libspqr.4`, `librbio.4` are empty placeholders of the same names, built by `coupling/tools/prune_bundle.py` (SuiteSparse_jll's init opens every SuiteSparse library; the coupling never calls these, as it uses EquivariantTensors' `nullspace_solver = :dense`) | MIT (ace-jax glue; no SuiteSparse code) | `LICENSE` |
| GCC runtime: libgcc_s, libgfortran, libgomp, libatomic, libssp, libstdc++ | CompilerSupportLibraries_jll 1.5.5+2 | `libgcc_s.1.1`, `libgfortran.5`, `libgomp.1`, `libatomic.1`, `libssp.0`, `libstdc++.6` | GPL-3.0-or-later WITH GCC-exception-3.1 | `licenses/GPL-3.0.txt`, `licenses/GCC-exception-3.1.txt` |
| libquadmath (GCC) | CompilerSupportLibraries_jll 1.5.5+2 | `libquadmath.0` | LGPL-2.1-or-later | `licenses/LGPL-2.1.txt` |
| GNU libiconv / libcharset (Julia artifact, loaded by Libiconv_jll's init) | 1.18 (Libiconv_jll 1.18.0+0) | `share/julia/artifacts/.../libiconv.2`, `libcharset.1` | LGPL-2.1-or-later (library) | `licenses/Libiconv.txt`, `licenses/LGPL-2.1.txt` |
| OpenSpecFun (Julia artifact, loaded by OpenSpecFun_jll's init) | 0.5.6 (OpenSpecFun_jll 0.5.6+0) | `share/julia/artifacts/.../libopenspecfun.2.1` | MIT (Faddeeva) AND public domain (AMOS) | `licenses/OpenSpecFun.txt` |

**No GPL-only code is bundled.** The GPL-2.0-or-later SuiteSparse modules
(UMFPACK, SPQR, RBio and the GPL parts of CHOLMOD) are replaced by the
placeholders above; `coupling/tools/check_bundle.py` fails a bundle that holds
any real copy. The copyleft that remains is (a) LGPL libraries, each shipped as a
separate, replaceable shared library, and (b) the GCC runtime under GPL-3.0 with
the GCC Runtime Library Exception, which permits its use in non-GPL works. GMP is
used under its LGPL-3.0-or-later option. The package metadata declares the whole-wheel
SPDX expression (`coupling/python/pyproject.toml`); `LICENSE` summarises it.
Source for every component is available from the upstream projects linked in
`licenses/Julia-THIRDPARTY.md`; the EquivariantTensors source is the fork
commit reported by `ace_jax_coupling.build_info()`.
