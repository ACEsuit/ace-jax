# ace-jax-coupling

The SO(3) coupling tables that [ace-jax](https://github.com/ACEsuit/ace-jax)
needs to author a *new* ACE basis shape, computed by
[EquivariantTensors.jl](https://github.com/ACEsuit/EquivariantTensors.jl)
compiled ahead of time with `juliac --trim=safe`. The wheel contains a
trimmed Julia runtime plus the compiled library; **no Julia installation is
needed**, and nothing is downloaded at runtime.

- Install: `pip install ace-jax-coupling`; ace-jax installs it as a dependency on these platforms.
- Platforms: Linux x86_64 / aarch64 (manylinux_2_28), macOS arm64 (11+), Windows x64.
- `ace_jax_coupling.build_info()` reports the EquivariantTensors source
  (repository + commit), Julia and JuliaC versions of the build.
- `ACEJAX_COUPLING_LIB=<bundle>/lib/libetcouple.<so|dylib>` points the package
  at a locally built library (development).

API: `couple_raw(mb_spec, Rnl_spec, Ylm_spec) -> RawCoupling` (0-based index
arrays; see ace-jax `docs/coupling-etshim-spec.md`). Licences of the bundled
components: `THIRD_PARTY_NOTICES.md`.
