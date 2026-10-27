# PACE reference tooling

Generates the committed PACE references under `fixtures/pace/`. The everyday
test suite needs neither toolchain; only regenerating the references does
(and the `pace-parity` CI job).

## Licences

The ace-jax package (`src/ace_jax`, MIT) neither includes nor imports the code
these tools use. Each tool fetches that code from ICAMS (or LAMMPS) under its own licence:

- **python-ace / pacemaker** (section 2) is under the
  [Academic Software Licence](https://github.com/ICAMS/python-ace/blob/master/LICENSE.md)
  (ASL): academic, non-commercial use only. Installing and running it here, and in the
  `pace-parity` CI job, is subject to the ASL. Commercial use needs a licence from
  ACEworks GmbH. The committed fixtures are its output (random-coefficient `.yace`
  files and the energies, forces and stresses it computes), not its code.
- **ML-PACE** (`lammps-user-pace`) is GPL (GPL-2.0 or GPL-3.0). `unit_ref.cpp` and
  `tabulated/` (`tab_eval.cpp`, `tab_time.cpp`, `CMakeLists.txt`) build against it, and
  `tabulated/0001-tabulated-radials.patch` modifies it: these files are distributed
  under ML-PACE's GPL terms, not the repository's MIT licence.
- **LAMMPS** is GPL-2.0. The patches in `fast/` modify its `pair_pace_kokkos` and are
  distributed under GPL-2.0.

## 1. Unit references (C++, no Python bindings)

`unit_ref.cpp` links the ML-PACE evaluator sources directly and dumps g_k(r)
for every radial family and inner regime, the core repulsion, ZBL,
`cutoff_func_poly`, `Fexp`/`FexpShiftedScaled` and the complex Y_lm.

- Sources: `lammps-user-pace` checked out at `~/gits/lammps-user-pace`
  (override with `-DPACE_SRC=...`). The committed references were made at
  ICAMS commit `99aa6e6`; CI builds `jameskermode/lammps-user-pace`
  `tabulated-radials` (ICAMS/lammps-user-pace#31: `v.2025.12.4.p1` plus the ML-PACE additions ace-jax's
  `.yace` export needs, see section 3). The commit is recorded in
  `unit_ref.npz` as `pace_commit`.

```bash
cmake -S pace_ref -B pace_ref/build -DCMAKE_BUILD_TYPE=Release
cmake --build pace_ref/build --target unit_ref -j
pace_ref/build/unit_ref | uv run python pace_ref/json_to_npz.py
```

## 2. Model references (python-ace)

python-ace has no PyPI wheels; it is built from source into its own venv
(`pace_ref/.venv`, git-ignored), never into ace-jax's.

```bash
uv venv pace_ref/.venv --python 3.11
uv pip install --python pace_ref/.venv/bin/python numpy ase pyyaml "git+https://github.com/ICAMS/python-ace"
pace_ref/.venv/bin/python pace_ref/make_fixtures.py
```

`make_fixtures.py` writes `fixtures/pace/{name}.yace` and `{name}_ref.npz`
(E/F/stress at the shipped `deltaSplineBins` and at 1e-4); see the plan,
`docs/dev/pace-yace-plan.md`, Task 2.

### Notes on the pyace fixture config (python-ace 0.2.7+192.g66c35ea, `~/gits/python-ace` @ 66c35ea)

- pyace defaults `inner_cutoff_type` to `distance`; `bond()` sets `density` explicitly.
- `prehc` is also the ZBL prefactor; the `zbl` fixture sets `core-repulsion: [1, 1]` so ZBL is active.
- Per-centre function overrides use pacemaker's unary key (`"Ge"`), not a tuple.
- `all_coeffs` is a settable property (there is no `set_all_coeffs`).
- On a flaky network, clone python-ace locally and `uv pip install ~/gits/python-ace`
  with `UV_HTTP_TIMEOUT=300`.

## 3. ML-PACE on ace-jax exports (`tab_eval`)

`tabulated/tab_eval.cpp` evaluates configurations from a `.yace` with ML-PACE's
product or recursive evaluator (`tab_eval model.yace [product|recursive]`).
`tests/test_pace_mlpace_yace.py` runs it on multi-element `write_yace` exports,
expanded and `summed=True`, against ace-jax. Those files need two ML-PACE
additions, `radbasename: Tabulated` and `summed_neighbour_A`, both on the
fork branch above; upstream `99aa6e6` fails every case.

```bash
cmake --build pace_ref/build --target tab_eval -j
ACEJAX_PACE_TAB_EVAL=pace_ref/build/tab_eval uv run pytest tests/test_pace_mlpace_yace.py
```
