# PACE reference tooling

Generates the committed PACE references under `fixtures/pace/`. The everyday
test suite needs neither toolchain; only regenerating the references does
(and the `pace-parity` CI job).

## 1. Unit references (C++, no Python bindings)

`unit_ref.cpp` links the ML-PACE evaluator sources directly and dumps g_k(r)
for every radial family and inner regime, the core repulsion, ZBL,
`cutoff_func_poly`, `Fexp`/`FexpShiftedScaled` and the complex Y_lm.

- Sources: `ICAMS/lammps-user-pace` checked out at `~/gits/lammps-user-pace`,
  commit `99aa6e6` (override with `-DPACE_SRC=...`). The commit is recorded in
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
`docs/pace-yace-plan.md`, Task 2.
