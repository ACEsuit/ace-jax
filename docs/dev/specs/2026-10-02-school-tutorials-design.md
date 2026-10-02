# MLIP-school notebooks as ace-jax tutorials

**Date:** 2026-10-02. **Status:** draft for review. **Branch:** `feat/school-tutorials`.

## Goal

Turn the six MLIP-school-2026 workshop notebooks (ACEsuit/MLIP-school-2026,
`notebooks/source/{e1,e1x,c,e2,e3,d}`) into ace-jax tutorials. They must run
locally with `uvx marimo edit --sandbox <url>`, with no cloud services. Each
tutorial renders as a site page through `docs/build_tutorials.py`, exactly as
tutorials 1 and 2 do.

The school notebooks called two servers:

- an **oracle** that labelled structures with MACE-MH-1;
- a **workbench** that fitted and evaluated with Julia ACEpotentials.

Both go:

- ace-jax replaces the workbench;
- a local, MIT-licensed MACE foundation model replaces the oracle.

## Decisions (agreed)

- **Labeller: MACE-MPA-0 (medium), MIT.** It is the most accurate MIT model for
  materials in ACEsuit/mace-foundations, trained on MPtrj + sAlex at PBE(+U). For
  Si and GaAs the +U correction does not apply.
  - All MH-1 labels are dropped, including E1x's shipped reference set, which is
    relabelled. Every tutorial then has one "truth".
- **Labels at render time, and in CI, come from shipped precomputed files.**
  - A live MACE cell runs only when a student leaves the default path, or brings
    their own system (D).
  - The docs build and CI therefore never install torch.
- **E1x shows both fits side by side.**
  - Plain least squares overfits as the basis grows, which is the school's curve.
  - The evidence fit flattens that curve.
  - The log-evidence then picks the basis size: overfitting plus Bayesian model
    selection.
- **C compares two MIT labellers, MACE-MPA-0 and MACE-MP-0b3.** They share a
  functional family but differ in training data and architecture. The lesson
  becomes that the reference is itself a model choice, and the ACE fit tracks
  whichever labeller it was taught. The size of the gap is re-measured on Si(111).
- **Scope: all six notebooks, in two PRs.**
  - PR 1: E1, E1x and C. These need fixed structures only.
  - PR 2: E2, E3 and D. These need on-demand labels.

## Tutorials

Tutorials 1 and 2 already exist (first fit, learned radials). The school
notebooks become tutorials 3–8. Each states what it assumes from the earlier
ones.

| # | From | Page | Teaches | Labels |
|---|---|---|---|---|
| 3 | E1 | Your data, your property | Build a strain/rattle set, fit, and check R²; a vacancy formation energy then exposes what RMSE hides | fixed set; the sliders snap to a precomputed grid |
| 4 | E1x | Basis size and overfitting | Train/test RMSE against basis size for least squares and for the evidence fit; the log-evidence as a model-selection criterion | shipped set (250 frames, relabelled) |
| 5 | E2 | Surfaces | Coverage gaps (bulk-only model against slabs), descriptor-space novelty, a repair set, and relaxations that diverge, then recover | defaults precomputed; live MACE otherwise |
| 6 | E3 | Automating curation | MD-driven active learning, random against novelty selection, at a fixed label budget; ace-jax's UQ as a third selector | live MACE; the canonical seeded run precomputed |
| 7 | C | The truth about the truth | γ(111) under two labellers; the fit follows its teacher | fixed (~15 structures × 2 models) |
| 8 | D | Bring your own data | Target property, dataset design, fit, and a coverage check for the user's own system | live MACE; GaAs demo precomputed |

Tutorial numbers follow the teaching order. The PR order does not change it.

## Shared machinery

- **Labels: `ace_jax.tutorials.labels`.** This is a small tutorial-support module,
  not stable API.
  - `label(structures, model="mpa-0")` returns `Atoms` carrying `energy`, `forces`
    and `virial`.
  - It first looks each structure up in a content-hash cache of shipped labels,
    keyed on the rounded positions, numbers, cell and pbc.
  - On a miss it runs the MACE model if `mace-torch` is importable.
  - Otherwise it raises an error naming the install line, and noting that the
    default path never needs it.
  - The cache files are extxyz files under
    `docs/user/tutorials/data/school/<tutorial>/labels-<model>.xyz`. Notebooks fetch
    them from GitHub, as tutorials 1–2 fetch their data.
  - A "labels used" counter keeps the school's labels-as-cost lesson, as a
    voluntary budget.
- **Label generator: `docs/user/tutorials/data/school/make_labels.py`.** It
  enumerates each tutorial's default structures, plus the slider grids for E1/E2,
  and labels them with MACE.
  - It runs in its own environment (mace-torch, CPU torch). That environment is
    not an ace-jax dependency.
  - Its output is checked in, with a provenance README: model, version, and an
    MIT notice.
- **Optional live MACE.** The notebooks' PEP 723 headers stay torch-free. The
  live path is documented as one extra install, with a pinned CPU torch index so
  that uv does not pull the CUDA wheels.
- **School machinery that is stripped.**
  - mograder `check`/`hint`, encryption, keys and the Pyodide client bootstrap.
  - Endpoints and API keys, server budgets, `load_model` chaining, and the Julia
    disclosure accordions.
  - Checks become `mo.callout` checkpoints, as in tutorial 1. Solutions become
    collapsible hints.
  - Cross-notebook models (for example C needing E2's model) are rebuilt in the
    notebook from shipped labels. A model is never resumed from a server.

## ace-jax additions

Each comes with its own tests.

1. **In-memory data.** `load_fit_data(cfg, train=[Atoms...], test=[...])` accepts
   lists of `ase.Atoms` with labels in `info`/`arrays`, as well as paths. The
   tutorials then need no temporary files.
2. **Stress labels.** MACE and ASE produce `stress`. The readers accept a stress
   key and convert it to the virial, `virial = −stress · volume`, so a fit can
   consume MACE labels directly.
3. **Evidence on the result.** `FitResult` exposes the MAP log-evidence. It is
   already computed and logged, but not returned. E1x plots it against the basis
   size.
4. **A plain least-squares reference fit, for E1x.** The plan measures first
   whether fixing the prior scale (`sigma_c`) very large, with no smoothness
   prior, reproduces the school's overfitting curve through the existing pipeline.
   If it does, that recipe is documented in the tutorial. If not, a minimal
   `FitConfig(solver="lstsq")` is added.

## Numbers and checks

Every threshold and reference number in the school notebooks was calibrated on
ACEpotentials' least-squares (QR) fit, with MH-1 labels. Examples are E1's
R² > 0.99 and the ~0.6 eV vacancy error, E1x's minimum at total degree 14, the
E2 tolerances, and E2's 24× novelty ratio. All of them are re-measured with
ace-jax and MPA-0 labels before the checkpoints are written. A checkpoint
asserts a robust qualitative outcome with margin; it never asserts a number
copied from the school.

## Testing and CI

- **Docs build.** It renders every tutorial on the default path from shipped
  labels, with no torch. A cell that raises fails the build, as now. Each
  tutorial stays a few CPU minutes; total build time is measured and budgeted.
- **Unit tests.** For the labels cache (hashing, lookup, the counter, the error
  without MACE) and for each ace-jax addition.
- **The live-MACE path.** It is exercised by `make_labels.py` itself, which runs
  locally and not in CI. The plan decides whether a small optional CI job that
  installs CPU torch is worth its cost.

## Out of scope

- HAL / `ase_uhal` (E3's optional take-home): it stays a pointer.
- The instructor dashboard, cohort tooling and WASM/Pyodide exports.
- An r²SCAN comparison: no MIT r²SCAN MACE model exists.
