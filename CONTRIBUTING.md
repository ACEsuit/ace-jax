# Contributing to ace-jax

Development uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ACEsuit/ace-jax && cd ace-jax
uv sync                              # the package (editable) + the dev group
```

`CLAUDE.md` is the maintainer's map of the code: layout, conventions and
pitfalls (written for coding agents, and useful to people too). Design specs,
plans and benchmark reports are in `docs/dev/`; releases follow `RELEASING.md`.

## Running the tests

```bash
uv run pytest                        # whole fast suite, 6 parallel workers (~2-4 min)
uv run pytest tests/test_efv.py      # a targeted run stays single-process
uv run pytest -m slow                # opt-in: real-model MCMC ladder, bit-exact driver goldens
```

A bare `pytest` uses pytest-xdist when it is installed (`ACEJAX_TEST_WORKERS`
sets the worker count; `-n 0` forces a single process, as CI does). The `slow`
marker is excluded by default. Tests needing an optional package (ace-jax-coupling,
lammps-jax, python-ace, sphericart, matscipy-neighbours, psutil) skip without
it; for the lammps-jax tests put a clone on the path
(`PYTHONPATH=<lammps-jax>/python`). See `CLAUDE.md` for the test switches.

## Linting and pre-commit

```bash
uv run ruff check                    # lint (config in pyproject.toml [tool.ruff])
uv run pre-commit install            # once per clone: ruff + whitespace/YAML/large-file hooks on commit
uv run pre-commit run --all-files    # what the `lint` CI job runs
```

No formatter is enforced: the code keeps its dense one-line style, and the ruff
rules that fight it (semicolon statements, short math names, import sorting,
line length) are off. Bump the ruff pin in the dev group and the `rev` in
`.pre-commit-config.yaml` together.

## Reference parity (maintainers / CI only)

The everyday test suite is pip-only, run against committed npz fixtures. Path-gated CI jobs regenerate the references and guard the seams to
the reference codes:

- **ACEfit parity** (`.github/workflows/julia-parity.yml`) — the design-matrix rows match ACEfit to 1e-8 and the linear
  solve matches ACEfit's `solve(QR)` to 1e-9, checked against the committed
  fixtures (`julia/export_model.jl`, `julia/acefit_qr_reference.jl`).
- **coupling-wheels** — builds the `ace-jax-coupling` wheels (manylinux_2_28
  x86_64/aarch64, macOS arm64, Windows x64), tests them in clean environments
  (matching upstream EquivariantTensors: exact indices, values within 4 ulp),
  and runs the coupling parity
  against the ACEpotentials references (`julia/coupling_reference.jl`).
- **prior-parity** — the smoothness prior (`basis/prior.py`) matches
  ACEpotentials' `algebraic_smoothness_prior` bit-for-bit
  (`julia/smoothness_reference.jl`), and the committed fixtures match a fresh run.
- **pace-parity** — the PACE path against the ML-PACE C++ (pinned
  `lammps-user-pace`) and python-ace, built from source in their own
  environments; see `pace_ref/README.md`.
