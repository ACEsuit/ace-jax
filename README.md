# ace-jax

Fit and evaluate **Atomic Cluster Expansion (ACE)** interatomic potentials in
pure **Python/JAX** — no Julia needed to fit or run.

- **Evaluate** exported ACE models (energy / forces / stress) and site descriptors.
- **Fit** the linear (M=0) model with stable solvers (Cholesky / QR / streaming
  QR / LSQR), reproducing ACEfit to machine precision.
- **Hybrid GP** fits with a calibrated uncertainty ladder (MAP → Laplace → VI →
  NUTS) and predictive `energy_std` / `forces_std`.
- **ASE calculator** (`ACECalculator`, `GPCalculator`).

## Install

```bash
pip install ace-jax           # core: evaluate + linear fit + ASE calculator
pip install ace-jax[gp]       # + GP/UQ hyperparameter ladder
pip install ace-jax[cuda]     # + CUDA 12 JAX
```

No Julia required. A model **definition** (basis + splined radials) is exported
once from [ACEpotentials.jl](https://github.com/ACEsuit/ACEpotentials.jl) to an
`.npz`; ace-jax consumes it. Two lifecycles:
- **use / fit / evaluate an existing model** → only the `.npz` (no Julia);
- **design a new basis** → one Julia export (`julia/export_model.jl`).

Models come as **unfitted definitions** (coefficients to be fit here) or
**fitted potentials** (ready to evaluate). A small model zoo is documented under
`docs/`; large artifacts are hosted as release assets, not committed.

## Quickstart

```python
import ace_jax as aj
model, meta = aj.load("si_fitted.npz")
from ace_jax import ACECalculator          # ase extra
atoms.calc = ACECalculator("si_fitted.npz")
atoms.get_potential_energy(); atoms.get_forces()
```

CLI: `ace-jax --help` (fit / gp-fit / eval / predict).

## Julia parity (maintainers / CI only)

The design-matrix rows match ACEfit to 1e-8 and the linear solve matches
ACEfit's `solve(QR)` to 1e-9. These are checked against **committed npz fixtures**
so the test suite runs pip-only. A path-gated CI job installs Julia and
regenerates the fixtures (`julia/export_model.jl`, `julia/acefit_qr_reference.jl`)
to guard against drift.
