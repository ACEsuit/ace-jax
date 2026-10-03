# School tutorials, PR 2 (E2, E3, D): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or
> subagent-driven-development) to implement this plan task by task. Steps use
> checkbox (`- [ ]`) syntax.

**Goal:** port the MLIP-school notebooks E2 (surfaces), E3 (automating curation) and
D (bring your own data) as ace-jax tutorials. They run locally, with shipped MACE
labels on every default path and live MACE only off it.

**Architecture:**
- **Workbench replacements.** The school's workbench calls become plain ace-jax and
  ASE code in the notebooks:
  - `fit_ace` becomes `fit`;
  - `evaluate` and `ace_validate` become `ACECalculator` plus the local γ;
  - `ace_relax` becomes ASE `BFGS`;
  - `ace_md` becomes ASE `Langevin` with `ACECalculator`;
  - `get_descriptors` becomes `site_descriptors` on a fixed reference basis.
- **Label sources.** `ace_jax.tutorials.labels` (PR 1) serves the labels. A few new
  deterministic builders go in `ace_jax.tutorials.structures`.
- **E3 is the one model-dependent path.** Its MD pools depend on the fitted models,
  and MD is chaotic across platforms. So the default path reads shipped pools, every
  frame of them labelled, instead of regenerating them.

**Tech stack:** ace-jax (this repo), ASE, marimo; mace-torch only in the offline
labelling environment.

**Spec:** `docs/dev/specs/2026-10-02-school-tutorials-design.md` (approved). PR 1 is
`docs/dev/plans/2026-10-02-school-tutorials-pr1.md`. Its machinery, its rulings
(evidence fit, measured checkpoints, one output per cell) and its file layout carry
over.

## Global constraints

- **Labels.**
  - Labeller: MACE-MPA-0 (`mpa-0`, MIT), float64, CPU.
  - Labels on every default path ship under `docs/user/tutorials/data/school/<e2|e3|d>/`.
  - The docs build and CI never install torch.
  - A live MACE cell runs only off the default path. Without mace-torch it raises
    `LabelsUnavailable` with the `EXTRA` install line.
- **Checkpoints.** Every number and checkpoint is measured with ace-jax and MPA-0 and
  set with margin. None is copied from the school (its 24× novelty ratio, its 0.02 and
  0.03 eV/Å² tolerances and its divergence story were measured on MH-1 with QR).
- **Runtime.** Each notebook runs in "a few CPU minutes", under about 5. E3 may run to
  about 8 if measured and stated, as tutorial 5 (E1x) does.
- **Output cells.** One output per cell: the page converter keeps only the first
  element of `mo.vstack`.
- **Numbering.** After the multi-element tutorial (3) lands, the school tutorials are
  4 (E1), 5 (E1x), 6 (E2), 7 (E3), 8 (C) and 9 (D). This plan writes 6, 7 and 9. If
  this PR lands before the multi-element one, use 5, 6 and 8, and the renumbering PR
  shifts them.
- **Notebook headers:** `ace-jax>=0.1.0` (PyPI). The school notebooks need
  `ace_jax.tutorials` from the installed package. This PR's additions to it
  therefore need a 0.1.1 release before the notebooks' `uvx` launch works from PyPI;
  the docs build uses the checkout.

## Review focus

1. **E3 default path, on any platform.** The selection reads shipped pools. If a
   pick differs from the offline run (float noise in descriptor distances), the label
   must still hit, because every pool frame is labelled. The refits then differ
   slightly, and the checkpoints must tolerate that.
2. **E2 relaxed γ against the labeller's own relaxed γ.** Never label an
   ACE-relaxed geometry on the default path: it is model-dependent. MACE's relaxed
   slabs ship instead.
3. **D upload path.** An uploaded `.xyz`, `.cif` or `.vasp` with 1 to 2 species and
   at most 64 atoms works only with mace-torch. It needs a clear `LabelsUnavailable`
   message, not a traceback, and the GaAs demo must stay the default.
4. **Slider grids.** E2's layers × vacuum grid snaps to shipped settings. A
   gap-check failure (too little vacuum) must be reported before any labelling.
5. **Novelty selector edge cases.** Duplicate scores, a pool smaller than the take,
   and empty pools after the physicality filter.

---

### Task 1: Structures for E2, E3 and D (`ace_jax.tutorials.structures`)

**Files:**
- Modify: `src/ace_jax/tutorials/structures.py`
- Test: `tests/test_tutorial_structures.py`

**Produces:**
- `E2_LAYERS = (4, 6, 8, 10, 12)`, `E2_VACUA = (4.0, 6.0, 8.0, 10.0, 12.0)`
  (Å per side) and `E2_MILLER = ((1,0,0),(1,1,0),(1,1,1))`.
- `e2_slabs(layers, vacuum) -> [Atoms]`, built with `slab(m, layers, vacuum)`
  (PR 1's shuffle-plane slab).
- `e2_repair_layers(layers) -> int`: the repair thickness. It is 4 unless `layers`
  is 4, in which case it is 6. This fixes the school's hard-coded-4 quirk.
- `e2_displaced(repair_slabs) -> [Atoms]`: two rattles per slab (0.05 and 0.12 Å,
  seed `5000 + 10*i + r`), tagged `slab-displaced`, with `miller` kept.
- `min_coordination(structures, cutoff=2.8)` and `min_vacuum_gap(structures)`.
- `e3_seed()`: 8 bulk cells at `linspace(0.94, 1.06, 8)`, rattle 0.02, seed 3000+i.
  It is the same list as `c_structures()["recipe"][:8]`, so it is labelled already.
- `e3_targets()`: the bulk and the 6-layer (111) slab, i.e. C's reference pair.
- `e3_md_starts()`: target bulk, the distractor (110) 6-layer slab and the target
  (111) slab.
- `d_system()`: GaAs zincblende, a = 5.65, cubic (8 atoms).
- `d_isolated(species)`: one atom in a 15 Å periodic box per species.
- `d_targets(system)`: the bulk plus the (100) 4-layer slab with 8 Å vacuum, tagged
  with `role`.
- `d_training(system, strain, rattle, n_train)`: `linspace(1-s, 1+s, n)` cell scales
  and rattle seed 2026+i.
- `d_repair(system)`: (100) slabs of 3 and 5 layers × 15 rattles in `linspace(0, 0.12)`,
  seed `4096 + 100*i + j`.
- `D_STRAINS`, `D_RATTLES` and `D_NTRAIN`: the grids the D sliders snap to.
- `structure_fingerprint(atoms)`: D's hold-out guard, invariant to permutation and
  wrapping.

- [ ] **Step 1: Failing tests.**
  - Determinism: building twice gives identical positions.
  - `e2_repair_layers(4) == 6` and `e2_repair_layers(6) == 4`.
  - `min_coordination` is 1 on a glide-cut (111) slab and at least 2 on the shuffle
    slabs.
  - `min_vacuum_gap(e2_slabs(6, 6.0)) >= 12 - 1e-6`.
  - `e3_seed()` equals `c_structures()["recipe"][:8]` by `structure_key`.
  - The fingerprint is invariant under permutation and wrapping, and differs for a
    different cell.
  - The D training set shares no fingerprint with `d_targets`.
- [ ] **Step 2:** run the tests and see them fail (missing names).
- [ ] **Step 3:** implement, in the existing style of `structures.py`.
- [ ] **Step 4:** run `pytest -n 0 tests/test_tutorial_structures.py tests/test_tutorial_labels.py`. Expected: PASS.
- [ ] **Step 5:** commit `feat(tutorials): structures for the surfaces, curation and bring-your-own-data tutorials`.

### Task 2: Descriptors, MD pools and selection helpers (`ace_jax.tutorials.campaign`)

**Files:**
- Create: `src/ace_jax/tutorials/campaign.py`
- Test: `tests/test_tutorial_campaign.py`

**Produces:**
- `reference_basis(elements=("Si",))`: `build_basis(BasisSpec(order=3, max_degree=10, rcut=5.5, elements=...))`.
  For Si that is 120 functions, matching the school's reference basis, and it is cached.
- `atom_descriptors(structures, basis_or_model) -> [ (n_atoms, D) ]`: per-structure
  `site_descriptors` rows.
- `md_pool(model_file, starts, *, temperature=400.0, dt_fs=1.0, n_steps=60, every=4, seed)`.
  It runs ASE `Langevin` (friction 0.01/fs) with `ACECalculator(model_file)` and
  returns the frames.
- `physical(frame, min_sep=1.7)`: the MIC minimum-separation filter.
- `nn_ratio(train_rows, query_rows)`: median query→train NN distance divided by the
  median train NN spacing (spacing clamped at 1e-12). It uses the Gram-form distances.
- `score_novelty(pool_rows, train_rows)`: per structure, the max over its atoms of the
  min distance to the training atoms (the school's score).
- The student-facing selectors stay in the notebooks: `pick_random` and `pick_novel`
  (Q1), plus a third, `pick_uncertain` (Task 6).

- [ ] **Step 1: Failing tests.**
  - The school's selector fixture: training clouds at 0, 0.01 and 0.02 plus an
    outlier at 25; the pool offsets 0.02, 3, 25.05 and 12. `score_novelty` gives
    approximately [0.00, 7.30, 0.12, 29.35], so the order is [3, 1].
  - `nn_ratio` is 1 for the same set, and scales linearly with the offset of a
    translated query.
  - `physical` rejects a frame with two atoms 1.0 Å apart.
  - `md_pool` on 8-atom Si with the tutorial-1 fixture model: same seed gives the same
    frames on one machine; the frame count is `n_steps // every + 1` per start; energy
    is finite.
  - `reference_basis()` has `len_basis == 120`.
- [ ] **Step 2–4:** fail, implement, then pass: `pytest -n 0 tests/test_tutorial_campaign.py`.
- [ ] **Step 5:** commit `feat(tutorials): descriptors, MD pools and novelty scores for the curation tutorials`.

### Task 3: Spike: E2's story under ace-jax and MPA-0 (measure before writing)

No production code; a scratch script.
1. Train the E1-default bulk set (strain 0.08, rattle 0.02: in the e1 cache) at
   order 3, degree 10, rcut 5.5 with the evidence fit. Measure:
   - γ(100), γ(110) and γ(111) against MPA-0 at the default slabs (6 layers, 8 Å);
   - the novelty ratio of slab atoms against bulk atoms (reference basis).
2. Add the repair slabs (4 layers), refit, and measure the γ errors.
3. Relax the default slabs with the repaired model (`BFGS`, fmax 0.03, 200 steps), with
   ASE. Record convergence, ΔE and the final force.
4. Add the displaced slabs, refit, and relax again. Compare the relaxed γ with MPA-0's
   own relaxed γ (relax the slabs with MACE, the same protocol).
5. Record everything in the ledger. **Rulings to make from the numbers:**
   - Does the "relaxations diverge after the surface repair" lesson reproduce?
     - If yes, keep the school's arc.
     - If no, because the evidence fit's prior keeps them stable, rule the lesson as
       "what the evidence fit buys", in the same way as tutorial 5's ruling, and
       frame step 5 as checking that the relaxed γ is still right.
   - The checkpoint thresholds, with margin.
   - The slider grid needed: whether vacuum 4 Å (an 8 Å gap) is in the grid as the
     failing case.

### Task 4: Spike: E3's campaign and the UQ selector

1. Run the two-driver campaign (random and novelty; 2 rounds × 4 picks; MD at 400 K,
   60 steps, every 4) from the E1-default model, offline with live MPA-0 (the mace
   environment). Record:
   - γ(111) error against labels for both drivers;
   - the mean picked index;
   - the runtime.
2. **Third selector: ace-jax's UQ.** Fit with `FitConfig(uq="ard", ...)` and pick by
   the per-structure max `forces_std` (`ACECalculator(model, posterior=...)`).
   Measure the fit cost and whether it beats random. If ARD costs too much at this
   size, try the GP arm (`m_per_species` 6–20, `GPCalculator` `energy_std`). Rule
   between them, ledgered.
3. Measure the seed variance: rerun with three rng seeds and record the spread. The
   school did not assert a winner, and neither will we, unless one wins with margin
   in every seed.
4. Decide what ships: for each driver and round, the MD pool frames (all labelled), as
   `e3/pools-<driver>-r<k>.xyz` plus `e3/labels-mpa-0.xyz`.

### Task 5: Labels for E2, E3 and D (`make_labels.py e2 | e3 | d`)

**Files:**
- Modify: `docs/user/tutorials/data/school/make_labels.py` and `README.md` (table and provenance).
- Create: `docs/user/tutorials/data/school/{e2,e3,d}/…`.

**What each mode labels:**
- `e2` labels:
  - every `e2_slabs(L, V)` on the grid, plus the reference bulk;
  - the repair slabs at 4 and 6 layers for every vacuum;
  - `e2_displaced` of each;
  - MACE-relaxed default slabs and their energies (the "relaxed truth"; written with
    `write_cache`).
- `e3` runs Task 4's final campaign offline, writes the pools and labels every pool
  frame. It needs ace-jax plus MACE in one environment (the existing `mace-labels`
  venv with `PYTHONPATH=src`).
- `d` labels, for GaAs:
  - the isolated atoms, the targets and `d_training` on the snapped slider grid;
  - `d_repair`.
- Sizes are recorded in the README. Keep each file under about 2 MB, and if E2's grid
  is too large, coarsen it and ledger that.

- [ ] **Step:** generate the labels. Verify zero cache misses on every default path
  with mace absent; the script is a test in `test_tutorial_labels.py` that loads each
  shipped file and checks the builders' structures against it, with no mace. Then
  commit `data(tutorials): MPA-0 labels for the surfaces, curation and GaAs tutorials`.

### Task 6: Tutorial 6, from E2: "Surfaces: coverage and repair"

**Files:** create `docs/user/tutorials/notebooks/school_surfaces_si.py`. Register it
(page `surfaces`) in `build_tutorials.NOTEBOOKS`, the nav, the index and `.gitignore`.

**Storyline:**
- **Title and run block:** builds on tutorials 4 and 1.
- **Step 1, the bulk model:** the E1-default fit, rebuilt from shipped labels.
- **Q1, build slabs.** The layer and vacuum sliders snap to the grid. Checks:
  - the vacuum gap is at least 12 Å, reported before labelling;
  - minimum coordination is at least 2 (the shuffle plane).
- **Ground truth:** the MPA-0 γ for the three faces, and the bulk model's γ. Show a bar
  chart and a parity plot (bulk on the line, slabs off it).
- **Q2, PCA coverage.** Reference-basis descriptors, a PCA scatter, and the
  `nn_ratio` novelty ratio, with a checkpoint threshold from Task 3.
- **Q3, the repair set:** `e2_repair_layers`. **Q4:** combine and refit. Checkpoints,
  from Task 3:
  - the γ errors shrink by a measured factor;
  - the test slabs are held out (by `structure_key`).
- **"Let it move":** relax with ASE `BFGS` and show the table. The narrative follows
  Task 3's ruling.
- **Q5, displaced slabs:** refit, relax, and compare with MPA-0's relaxed γ
  (checkpoint).
- **Q6, reflection** (accordion): the school's model answer (reconstructions, steps,
  temperature, uncertainty-driven curation).
- **Exercises, then the summary.**

- [ ] Measure at the defaults and at two grid points. Set the checkpoints, run the
  page build, then commit
  `docs(tutorials): tutorial 6, surfaces, coverage and repair (from MLIP-school E2)`.

### Task 7: Tutorial 7, from E3: "Automating curation"

**Files:** create `docs/user/tutorials/notebooks/school_curation_si.py`. Page `curation`.

**Storyline:**
- **Targets and seed.**
- **Q1, selectors.** Students write `pick_random` and `pick_novel`; the checks are the
  school's fixture. The third selector, `pick_uncertain` (Task 4's choice), is provided
  and explained.
- **The campaign.**
  - Default: read the shipped pools for each driver and round, select, label from the
    cache, and refit with the evidence fit.
  - Off the default path, a "run MD live" switch regenerates the pools with `md_pool`;
    it needs mace-torch for the new labels.
  - Show the γ(111) error against labels for the three drivers, and a table.
- **Q2, the mean picked index**, with the pool-order explanation.
- **Q3, HAL take-home:** a pointer to `ase_uhal`, ungraded.
- **Q4, reflection:** novelty is not importance; the descriptor distance is a proxy for
  uncertainty, and ace-jax's UQ selector is the direct version.
- **Checkpoints:** histories are finite and of equal length. A winner is asserted only
  if Task 4 shows one with margin in every seed.

- [ ] Measure the runtime. If it is over about 5 min, cut it to 3 starts × 10 frames
  or one round, and ledger the choice. Then commit
  `docs(tutorials): tutorial 7, automating curation (from MLIP-school E3)`.

### Task 8: Tutorial 9, from D: "Bring your own data"

**Files:** create `docs/user/tutorials/notebooks/school_byod.py`. Page `bring-your-own-data`.

**Storyline:**
- **Q1, declare the target:** a dict with name, quantity, units, tolerance and method.
- **Q2, the system:**
  - an upload widget (marimo `mo.ui.file`, read in memory); GaAs is the default;
  - checks: periodic, 1 to 2 species (an MPA-0 element set), at most 64 atoms;
  - a non-default system needs mace-torch, which is stated.
- **Q3, references and truth:**
  - E0 from the isolated atoms (`e0="model"`, with the E0s from labels);
  - the target γ(100) from the labeller.
- **Q4, the training recipe.** Sliders snapped to the D grids. The observation guard is
  dropped: the evidence fit needs no observations > basis size (the prior regularises).
  Say so, and show the basis size of the categorical two-species basis (`BasisSpec`
  elements = species) at degree 8.
- **Q5, measure:** the fitted target against the truth and the tolerance.
- **Q6, coverage:**
  - model-basis descriptors and PCA;
  - the `nn_ratio` against tutorial 6's measured silicon ratio.
- **Q7, one repair round:** `d_repair` and a refit with the same basis, then the error
  against cumulative labels.
- **Take-home:** `write_cache` or `ase.io.write` of the labelled data, the fitted
  `model.npz`, and `aj fit` command lines for the same run.

- [ ] Measure on GaAs, then commit
  `docs(tutorials): tutorial 9, bring your own data (from MLIP-school D)`.

### Task 9: Docs, the full check, and the build budget

- The index table, the nav and the README tutorial line.
- The data README.
- `CLAUDE.md` (`tutorials/campaign.py` and the new data directories).
- `SKILL.md`, only if new public options appear (none are planned).
- `CHANGELOG.md`, an unreleased section: `ace_jax.tutorials` additions.
- **Checks:**
  - pre-commit, plus the name guards and `test_tutorial_notebooks.py` (add the
    cycle-free check for each new notebook and for each exercise edit that the
    exercises suggest);
  - the strict docs build from scratch with mace absent, with its total time recorded
    against the 40-min CI timeout (budget: under about 30 min for all nine tutorials);
  - the full suite once.
- Then the final whole-branch review (fresh reviewer, strongest model), one fix pass,
  and a PR. Then cut 0.1.1, so the notebooks' `uvx` launch has the new
  `ace_jax.tutorials`.
