# Per-atom uncertainty on defect combinations (Cantor CrMnFeCoNi)

Which per-atom uncertainty score tells a user **where not to trust** an ACE model when simple
defects that were in the training set combine in ways that were not, the situation in big-cell
fracture and plasticity runs? This directory holds the benchmark, the fits and the scoring. It is
research code: it records how the results below were produced, and it is not a supported API.

Cantor is a proxy system. Labels come from **MACE-MH-1** (torch, head `matpes_r2scan`: the head
that reproduces the cantor4k labels to 5e-5 eV/atom). Set `MACE_MODEL` to the model file and
`ACEGP_DATA` to the data root (default `~/acegp-data`).

## Benchmark (`gen/`)

All cells are equiatomic with random occupancy and centred on **MACE's own Cantor lattice
constant**, a0 = 3.6502 Å.
- MACE's constants (`gen/big2.py`, `cantor_properties`) are C11/C12/C44 = 128/101/94 GPa and
  γ(111) = 2.2 J/m² unrelaxed.
- The original cantor4k set used a0 = 3.59 Å, which put its median pressure at **+7.3 GPa**.
- The re-centred set sits at +0.6 GPa.

| set | contents | split |
|---|---|---|
| `make_bulk.py` | 4000 bulk cells, the cantor4k recipe at a0 = 3.6502 (±3 % strain, 0.02 shear, 0.02–0.10 Å rattle) | 3200 train / 800 test |
| `make_defects.py`, train families | vacancy, (100) and (111) slabs, intrinsic stacking fault (pure Ni check: 80 mJ/m² unrelaxed) | 120 train / 30 test each |
| `make_defects.py`, OOD families | NN divacancy, vacancy in the surface layer, vacancy at the stacking fault, slab under 4–8 % tension | 100 each, held out |
| `big2.py` + `modal/modal_big2.py` | (111)[1-10] crack cylinders (R = 60 Å, 3236 atoms) at K/K_G = 1.1/1.2/1.3 with K_G from MACE's C_ij and γ; edge (R = 50 Å) and screw (R = 60 Å) a/2⟨110⟩ dislocations built dissociated; relaxed with the boundary fixed, plus 0.05 and 0.10 Å rattles | 30 configs, held out |

All six cracks stay open and lattice-trapped, with the tip within ±4 Å of the seed. MACE-MH-1 in
float64 fails on single cells above ~7k atoms (A100 out of memory, B200 illegal memory access), so
the cylinders were sized to 3–4k atoms.

## Fits (`modal/`)

`fit_bench.py` drives the `ace_jax.fit.pipeline` API on B200s (`modal_bench365.py`), using the
production basis `cantor_embed_d16_deg10` (L = 15 035).
- **POPS** (linear): 82 min.
- **GP** (PCA-128, host-cached LML, 4 L-BFGS starts): 5.2 h.

Test errors are E 3.1 meV/atom and F 0.069 eV/Å for both.

Big cells are not predicted through the pipeline. One >2.8k-atom cell's dense edge Jacobian
exceeds 2³¹ elements, which breaks the int32-indexed GEMM autotuning. Their per-atom errors come
from the saved linear model via `ACECalculator` (`big_errors`).

## Scoring (`scoring/`)

- `descriptors.py`: per-site descriptors for train, test, OOD and big sets.
- `score_atoms.py`: per-atom scores against |ΔF| per atom.
  - **Hyperparameters are chosen only on the in-distribution test split.**
  - Every OOD family and the big cells are held out.
- **Metrics:**
  - Spearman ρ(score, |ΔF|);
  - AUROC of OOD atoms against test atoms;
  - recall of the worst 5 % of atoms within the top 10 % of scores;
  - crack/dislocation core (< 10 Å) against far interior (> 25 Å).
- `bayes_bodyorder.py`:
  - Bayesian linear regression posteriors under four priors: the fitted Γ, isotropic, and ARD per
    body order (on Γ, or isotropic), each fitted by evidence;
  - nested 2/3/4-body truncations and Bayesian model averaging over them;
  - per-atom posterior σ of forces and of site energies.

## Results (2026-09-28)

AUROC is OOD atoms against test atoms; ρ is Spearman ρ(score, |ΔF|).

| score | ρ test | AUROC: divac / vac_surf / vac_sf / strained surf | ρ: divac / vac_surf / vac_sf / strained surf | ρ: crack / edge / screw |
|---|---|---|---|---|
| POPS σ_F (pipeline) | 0.32 | 0.68 / 0.53 / 0.70 / **0.27** | 0.26 / 0.02 / 0.22 / 0.08 | – |
| GP σ_F (pipeline) | 0.21 | 0.58 / 0.58 / 0.59 / **0.31** | 0.16 / 0.05 / 0.19 / 0.12 | – |
| **BLR posterior σ_F, ARD per body order** | 0.28 | **0.75 / 0.90 / 0.85 / 0.88** | 0.19 / 0.27 / 0.21 / 0.20 | (needs design rows) |
| descriptor leverage (ridge 1e-8) | 0.26 | 0.69 / 0.84 / 0.81 / 0.87 | 0.22 / 0.32 / 0.24 / 0.23 | 0.32 / 0.20 / 0.20 |
| kNN, whitened centred pair channels | 0.29 | 0.68 / 0.67 / 0.77 / 0.88 | 0.24 / 0.22 / 0.32 / 0.07 | **0.39 / 0.31 / 0.30** |
| body-order difference \|F₄ − F₃\| | 0.21 | 0.64 / 0.80 / 0.65 / 0.78 | 0.15 / 0.25 / 0.20 / 0.16 | – |
| T²+Q (PCA 128), full descriptor | 0.18 | 0.68 / 0.83 / 0.75 / 0.67 | 0.10 / 0.21 / 0.10 / 0.15 | 0.10 / 0.04 / 0.06 |
| DADApy k*NN density (whitened PCA 32, ID ≈ 21) | 0.08 | 0.57 / 0.55 / 0.69 / 0.35 | ≈ 0 | ≈ 0 |
| site-energy posterior σ (any prior) | −0.11 | 0.47 / 0.30 / 0.47 / 0.12 | < 0 | < 0 |

### Findings

1. **The pipeline's σ fails where fracture needs it.** POPS's hypercube σ and the GP's σ are
   *more* confident on strained surfaces than on the test set (AUROC 0.27 and 0.31). POPS also has
   the best in-distribution ρ (0.32), so choosing an estimator in-distribution picks the wrong one.
2. **The Bayesian posterior predictive σ of each atom's force matches or beats descriptor
   leverage** on every held-out combination family.
   - This is the Bayesian counterpart of the D-optimality extrapolation grade: γ = max|c_j| is an
     L∞ bound on the square root of a flat-prior BLR variance against the active set.
   - ARD per body order, fitted by evidence (+4865 nats over the fitted Γ prior), leaves the
     2-body terms nearly unregularised and shrinks the 4-body terms most.
3. **Per-atom UQ must be defined on observables.** Site energies are not identifiable from energy
   and force data. Their posterior σ tracks that gauge freedom and is anti-correlated with error.
   Leverage-style scores work only because they treat site energies as observed.
4. **Evidence-weighted averaging over body-order truncations collapses** onto the full model (a
   64 000-nat gap to 3-body), so its between-model term is zero. The raw body-order difference is
   weaker than the posterior σ.
5. **Sampling density on the full-descriptor manifold does not track error**, even though that
   manifold is low-dimensional (2NN intrinsic dimension 13–21). A density in the whitened pair
   (2-body) subspace does, and it is the best score so far on cracks and dislocations.
6. **The epistemic ARD σ has a constant scale.** It is ~7× too small, but its rms-z is 6.2–7.0 in
   every family, in distribution and held out. So one temperature fitted in-distribution (a
   tempered or generalised posterior) should transfer. POPS's σ is calibrated in-distribution
   (rms-z 0.6–0.7) but drifts OOD (strained surfaces 1.09).

Next: the calibrated tempered ARD posterior (coverage on held-out families), node-chunked design
rows (force σ on the big cells), and per-atom posterior σ_F as a calculator output.
