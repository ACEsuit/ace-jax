# ace-jax

Fit and evaluate **Atomic Cluster Expansion (ACE)** interatomic potentials in
pure **Python/JAX** — no Julia needed to fit or run.

- **Evaluate** exported ACE models (energy / forces / stress) and site descriptors.
- **Fit** the linear (M=0) model with stable solvers (Cholesky / QR / streaming
  QR / LSQR), reproducing ACEfit to machine precision.
- **Hybrid GP** fits with a calibrated uncertainty ladder (MAP → Laplace → VI →
  NUTS) and predictive `energy_std` / `forces_std`.
- **ASE calculator** (`ACECalculator`, `GPCalculator`).
- **Author** the symmetry-adapted coupling coefficients from Python via
  [EquivariantTensors.jl](https://github.com/ACEsuit/EquivariantTensors.jl) —
  bit-for-bit with ACEpotentials, no manual Julia setup (optional `authoring` extra).

## Install

```bash
pip install ace-jax             # core: evaluate + linear fit + ASE calculator
pip install ace-jax[gp]         # + GP/UQ hyperparameter ladder
pip install ace-jax[authoring]  # + Python basis coupling (EquivariantTensors via juliacall)
pip install ace-jax[cuda]       # + CUDA 12 JAX
```

No Julia is required to **use, fit, or evaluate** a model. A model **definition**
(basis + splined radials) is exported to an `.npz` that ace-jax consumes. The
model-authoring seam has three paths:

- **use / fit / evaluate an existing model** → only the `.npz` (no Julia);
- **build the SO(3) coupling table in Python** → the `authoring` extra
  constructs the `(n,l)` specification and the symmetry-adapted A→B coefficients
  directly via EquivariantTensors. `juliacall`/`juliapkg` auto-provision a
  pinned private Julia the first time (no manual install, and no full
  ACEpotentials stack). This is the coupling stage; radials / pair basis /
  embedding are still taken from a Julia export today (a follow-up closes that
  gap);
- **export a whole new basis from Julia** → the original path
  (`julia/export_model.jl`), still fully supported.

Models come as **unfitted definitions** (coefficients to be fit here) or
**fitted potentials** (ready to evaluate). A small model zoo is documented under
`docs/`; large artifacts are hosted as release assets, not committed.

## Quickstart

```python
import ace_jax as aj
model, meta = aj.load("si_fitted.npz")
from ace_jax import ACECalculator
atoms.calc = ACECalculator("si_fitted.npz")
atoms.get_potential_energy(); atoms.get_forces()
```

CLI: `ace-jax --help` (fit / gp-fit / eval / predict).

### PACE (pacemaker) potentials

`.yace` files load directly and evaluate in JAX (ASE calculator, lammps-jax):

```python
atoms.calc = ACECalculator("model.yace")
```

Supported: ChebExpCos / ChebPow / ChebLinear / SBessel radials, FinnisSinclair
and FinnisSinclairShiftedScaled embeddings, `density` / `distance` / `zbl` inner
cutoffs. `write_yace(model, spec, path)` writes a (possibly modified) model back.
Checked against the ML-PACE C++ and python-ace; see `docs/pace-yace-spec.md`.

## Authoring the coupling table in Python (EquivariantTensors)

With the `authoring` extra, `ace_jax.construct` builds an ACE basis's
symmetry-adapted coupling coefficients without a Julia export step:
`construct.spec.build_spec(...)` enumerates the admissible `(n,l)` many-body
specification (total-degree / `wL`), and `construct.coupling.couple(...)` bridges
to EquivariantTensors to produce the A→B symmetrisation matrix. `juliacall`
calls EquivariantTensors.jl directly — a far lighter dependency than the full
ACEpotentials stack — and `juliapkg` provisions the pinned Julia + ET
automatically, so you never install or manage Julia yourself.

The coupling is verified **bit-for-bit** against ACEpotentials for orders 2–4,
including the degenerate `nnll` blocks that need a subspace (row-space) match.
See `docs/coupling-etshim-spec.md` for the design and `tests/test_coupling_parity.py`
for usage.

## Julia parity (maintainers / CI only)

The everyday test suite is pip-only (no Julia), run against committed npz
fixtures. Two path-gated CI jobs guard the Julia-facing seams:

- **julia-parity** — the design-matrix rows match ACEfit to 1e-8 and the linear
  solve matches ACEfit's `solve(QR)` to 1e-9, checked against the committed
  fixtures (`julia/export_model.jl`, `julia/acefit_qr_reference.jl`).
- **coupling-parity** — the Python coupling coefficients match ACEpotentials
  bit-for-bit, regenerating references from the same pinned EquivariantTensors
  (`julia/coupling_reference.jl`).
