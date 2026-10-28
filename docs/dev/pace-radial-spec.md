# PACE radial, core and embedding functions: specification

This file defines the functions in `src/ace_jax/eval/pace_radial.py` as equations.
The module is implemented from this specification. Where a published equation exists,
it is cited. Where ML-PACE's output differs from that equation, or no equation is
published, the definition here is stated as the observed behaviour, marked
**matched to ML-PACE output**, and fixed by the parity data in `fixtures/pace/`.

## Why

ML-PACE (`lammps-user-pace`) is GPL; the ace-jax package is MIT. The first version of
`pace_radial.py` followed ML-PACE's C++ function by function. ICAMS (R. Drautz,
2026-10-28) advised that code written to match the published ideas and the results, not
derived from the C++, is our own, and pointed to the definitions in Bochkarev et al.
This specification is that route: the equations come from the papers, and ML-PACE
enters only as reference output (`fixtures/pace/unit_ref.npz` and the `.yace` parity
fixtures), which is data. `pace_ref/README.md` covers the licences of the reference
tools themselves.

## Sources

- **[PRM]** A. Bochkarev, Y. Lysogorskiy, S. Menon, M. Qamar, M. Mrovec, R. Drautz,
  *Efficient parametrization of the atomic cluster expansion*, Phys. Rev. Materials 6,
  013804 (2022), Appendix C, Eqs. (C1)–(C13) and Fig. 12.
- **[SI]** Y. Lysogorskiy et al., *Performant implementation of the atomic cluster
  expansion (PACE) and application to copper and silicon*, npj Comput. Mater. 7, 97
  (2021), Supplementary Information, Eqs. (15)–(22).
- **[Kocer]** E. Kocer, J. K. Mason, H. Erturk, J. Chem. Phys. 150, 154102 (2019),
  Sec. II.
- **[ZBL]** J. F. Ziegler, J. P. Biersack, U. Littmark, *The Stopping and Range of Ions
  in Matter* (Pergamon, 1985): the universal screening function.
- **[LAMMPS]** LAMMPS documentation, `pair_style zbl`: the switching function S(r).

## Status summary

| function | source | deviation from the source |
|---|---|---|
| `cutoff_func_poly` | [PRM] C8–C10 | none |
| ChebPow | [PRM] C3–C4, [SI] 16–17 | none |
| ChebExpCos | [PRM] C5–C6, Fig. 12a | g₀ is the envelope; cosine outer taper (§3.2) |
| SBessel | [PRM] C7, [Kocer] | none (evaluated by our own recurrence or matrix form) |
| ChebLinear | none published | matched to ML-PACE output |
| inner distance cutoff | [PRM] C2 | `zbl` regime: §4 |
| `radcore` | [PRM] C12, [SI] 19 | cosine envelope; inner cutoff only for `distance` (§5) |
| `pace_zbl` | [ZBL], [LAMMPS] | factor ½ (site split) and outer polynomial taper (§6) |
| `fexp_shifted_scaled` | [SI] 22 (m = ½) | general m: derived here (§7.2) |
| `fexp` | none published | matched to ML-PACE output (§7.1) |
| `hermite_table` | cubic Hermite interpolation | none |
| site energy | [PRM] C13 | `zbl` regime switch (§8) |

## 1. Notation

- r is the bond length, r_c the bond cutoff (`rcut`), Δ the outer taper width (`dcut`),
  r_in and Δ_in the inner cutoff and its width (`r_in`, `delta_in`), K the number of
  radial basis functions (`nradbase`), λ the radial scale (`radparameters[0]`).
- T_k is the Chebyshev polynomial of the first kind, evaluated by the three-term
  recurrence T_{k+1} = 2x T_k − T_{k−1}.
- Every function is zero outside its support. Every untaken branch is evaluated on a
  safe argument (a double `where`), so values and gradients stay finite at r ≥ r_c,
  r = 0, x = 0 and m = 1. This is an implementation requirement, not part of a
  definition.

## 2. Polynomial cutoff — `cutoff_func_poly(r, r0, Δ)`

[PRM] C8–C10, with r_c → r0:

    f(r) = 1                 for r ≤ r0 − Δ
         = (1 + y(x)) / 2    for r0 − Δ < r < r0
         = 0                 for r ≥ r0
    x = 1 − 2 (1 + (r − r0)/Δ),   y = a x + b x³ + c x⁵,   a = 15/8, b = −5/4, c = 3/8.

y(±1) = ±1 and y′, y″ vanish at x = ±1, so f is C². Δ = 0 is a step at r0.

## 3. Radial basis g_k(r) — `radbase`

All families return K functions on 0 < r < r_c; see §4 for the inner cutoff.

### 3.1 ChebPow

[PRM] C3–C4 (equivalently [SI] 16–17):

    g_k(r) = (1 − T_k(x)) / 2,   x = 1 − 2 (1 − r/r_c)^λ,   k = 1, …, K.

No outer taper is applied: for λ > 1, g_k and g_k′ vanish at r_c.

### 3.2 ChebExpCos

[PRM] C5–C6:

    x = 1 − 2 (e^{λ(1 − r/r_c)} − 1) / (e^λ − 1)
    e(r) = (1 + cos(π r/r_c)) / 2
    g_k(r) = (1 − T_k(x)) / 2 · e(r).

Deviations, matched to ML-PACE output:

- The set is g₀ = e(r) followed by g_k for k = 1, …, K − 1 (K functions). The
  formula gives zero at k = 0, so g₀ is the envelope alone, as plotted in [PRM]
  Fig. 12a.
- The outer taper over (r_c − Δ, r_c) is a cosine, not the polynomial f of [PRM] C2:

      g_k → g_k · (1 + cos(π (r − r_c + Δ)/Δ)) / 2   for r > r_c − Δ.

  All the parity fixtures use Δ = 0.01 Å, so this difference is visible in the last
  0.01 Å before the cutoff.

### 3.3 SBessel

[PRM] C7 ([Kocer] Sec. II), with s = π r/r_c:

    f_k(r) = (−1)^k √2 π (k+1)(k+2) / (r_c^{3/2} √((k+1)² + (k+2)²))
             · (sinc((k+1) s) + sinc((k+2) s))
    e_k = k² (k+2)² / (4 (k+1)⁴ + 1),   d_0 = 1,   d_k = 1 − e_k / d_{k−1}
    g_0 = f_0,   g_k = (f_k + √(e_k / d_{k−1}) g_{k−1}) / √d_k,   k = 0, …, K − 1.

K = 0 returns one function (g_0). The implementation computes sin(j s), j = 1…K+1,
either by a sin/cos rotation recurrence (one sin, one cos per edge) or as
sinc(j s) times a constant matrix that folds the f → g recursion (`sbessel_matrix`);
both are exact to roundoff (`tests/test_pace_radial_rec.py`).

### 3.4 ChebLinear

No published equation. **Matched to ML-PACE output:**

    g_k(r) = (1 − T_k(1 − r/r_c)) / 2,   k = 1, …, K.

This is not ChebPow at λ = 1 (whose argument is 2r/r_c − 1). No outer taper is
applied: g_k(r_c) = (1 − cos(kπ/2))/2 is zero only for k divisible by 4, and the
basis is cut to zero at r_c.

### 3.5 Tabulated

The values come from the `.yace` file's tables of g_k and dg_k/dr on a uniform grid
r_i = i h, interpolated by the cubic Hermite form (`hermite_table`):

    t = r/h − i,   g = h₀₀(t) g_i + h h₁₀(t) g′_i + h₀₁(t) g_{i+1} + h h₁₁(t) g′_{i+1},

with the standard Hermite basis h₀₀ = (1 + 2t)(1 − t)², h₁₀ = t(1 − t)²,
h₀₁ = t²(3 − 2t), h₁₁ = t²(t − 1).

## 4. Inner cutoff of the radial basis

[PRM] C2: in the `distance` regime,

    g_k(r) → g_k(r) · (1 − f(r; r_in, Δ_in)),

with f from §2, and g_k = 0 for r ≤ r_in − Δ_in. In the `density` regime the basis has
no distance cutoff (the energy-based cutoff of [PRM] C13 replaces it; it lives in
`pace_model.py`).

`zbl` regime, **matched to ML-PACE output**: the basis is not cut at the
`.yace` file's r_in. It is cut as in the `distance` regime at an effective
r_in′ = 1 Å when Δ_in = 0 (a hard step at 1 Å), and at r_in′ = 0 otherwise (no cut
for r > 0). The ZBL switch in `pace_model.py` uses the file's r_in and Δ_in.

## 5. Core repulsion — `radcore`

[PRM] C12 and [SI] 19 give φ_core(r) = w₀ e^{−w₁ r²}/r · f(r; r_in, Δ_in).
**Matched to ML-PACE output:**

    φ_core(r) = |w₀| e^{−|w₁| r²} / r · (1 + cos(π r/r_c)) / 2 · c(r),   0 < r < r_c,

with c(r) = f(r; r_in, Δ_in) (§2) in the `distance` regime and c(r) = 1 in the `density`
regime. The extra cosine envelope takes φ_core smoothly to zero at r_c. The absolute
values make φ_core positive for any sign of the stored parameters. φ_core = 0 for
r ≥ r_c.

## 6. ZBL core — `pace_zbl`

[ZBL] universal screening, with a = 0.46850 Å / (Z_i^{0.23} + Z_j^{0.23}) (0.8854 a_B)
and k_e = 14.399645351950543 eV Å:

    E(r) = φ(r/a)/r,   φ(x) = Σ_{i=1..4} c_i e^{−d_i x}
    c = (0.18175, 0.50986, 0.28022, 0.02817),   d = (3.19980, 0.94229, 0.40290, 0.20162).

With u_i = e^{−d_i r/a} and q_i = −d_i/a − 1/r, the derivatives are
E′ = Σ c_i u_i q_i / r and E″ = Σ c_i u_i (q_i² + 1/r²) / r.

[LAMMPS] switching over (r₁, r_c), r₁ = r_c − Δ, δ = r_c − r₁:

    A = (−3 E′(r_c) + δ E″(r_c)) / δ²
    B = (2 E′(r_c) − δ E″(r_c)) / δ³
    C = −E(r_c) + δ E′(r_c)/2 − δ² E″(r_c)/12
    S(r) = C for r ≤ r₁,   A (r − r₁)³/3 + B (r − r₁)⁴/4 + C for r₁ < r < r_c,

so E + S and its first two derivatives vanish at r_c.

The site contribution, **matched to ML-PACE output**, is

    φ_zbl(r) = w₀ (k_e / 2) Z_i Z_j (E(r) + S(r)) · f(r; r_c, Δ),   0 < r < r_c,

where w₀ is the stored prefactor. The factor ½ splits the pair energy between the two
site energies. The polynomial taper f (§2) over the same interval as S is an extra
factor.

## 7. Embedding functions

### 7.1 `fexp` (FinnisSinclair)

No published equation. **Matched to ML-PACE output:** a power law |x|^m that becomes
linear near x = 0, so F′(0) is finite for m < 1:

    F(x) = sgn(x) ((1 − g) |x|^m + λ g |x|),   g = e^{−(w|x|)³},   w = 10⁶,   λ = w^{1−m}.

The linear term λ|x| equals |x|^m at |x| = 1/w, where g ≈ 0.37. F(0) = 0 and
F′(0) = λ. 1 − g is evaluated as −expm1(−(w|x|)³), so the small-|x| side has no
cancellation.

### 7.2 `fexp_shifted_scaled` (FinnisSinclairShiftedScaled)

[SI] 22 gives the m = ½ case:

    F(x) = sgn(x) (√(|x| + e^{−|x|}/4) − e^{−|x|}/2).

The general-m form, **derived here** and matched to ML-PACE output, is

    F(x) = sgn(x) ((|x| + c e^{−|x|})^m − d e^{−|x|}),
    c = (1/m)^{1/(m−1)},   d = c^m.

- c is the point where d(x^m)/dx = 1, so the shifted power has unit slope at x = 0
  before the e^{−|x|} correction.
- d = c^m gives F(0) = 0.
- At m = ½: c = 1/4 and d = 1/2, which is [SI] 22.
- As m → 1, c → e^{−1} and F → x. The implementation uses F(x) = x for
  |m − 1| < 10⁻¹⁰, where the exponent 1/(m − 1) is ill-conditioned.

[SI] says this form gives F′(0) = 1. It does not: F′(0) = m c^{m−1}(1 − c) + d, which
is 1.25 at m = ½. The implementation follows the equation, not that sentence.

## 8. Site energy — `PACEModel._energy_tail`

The energy assembly is in `pace_model.py`, not in `pace_radial.py`. It uses §2, §5–§7.

- `density` and `distance` regimes, [PRM] C13, with ρ_core = Σ_j φ_core(r_ij) (C11) and
  the polynomial f of §2 on ρ_core:

      E_i = Σ_p w_p F_p(ρ_p) · f(ρ_core; ρ_c, Δ_c) + ρ_core + E0.

- `zbl` regime, **matched to ML-PACE output**: the switch uses the nearest neighbour.
  Let j* minimise r_ij − (r_in − Δ_in) over the bonds of atom i, with that bond's r_in
  and Δ_in, and let s = r_ij* − (r_in − Δ_in). Then

      f = f(Δ_in − s; Δ_in, Δ_in),   E_i = Σ_p w_p F_p(ρ_p) · f + ρ_zbl (1 − f) + E0,

  with ρ_zbl = Σ_j φ_zbl(r_ij) (§6). f = 1 for r_ij* ≥ r_in and f = 0 for
  r_ij* ≤ r_in − Δ_in. An atom with no neighbours has f = 1.

## 9. Tests

- `tests/test_pace_radial.py`: every function and regime against
  `fixtures/pace/unit_ref.npz` (ML-PACE output, `pace_ref/unit_ref.cpp`) to ≤ 1e-12
  relative; finite gradients at r ≥ r_c, x = 0 and Δ = 0.
- `tests/test_pace_radial_rec.py`: SBessel recurrence and matrix forms against the direct
  [PRM] C7 evaluation.
- `tests/test_pace_model.py`, `tests/test_pace_mlpace_yace.py`: whole-model energies,
  forces and stresses against python-ace and ML-PACE on the `.yace` fixtures, which
  cover every radial family, every inner regime and both embeddings.
