# Force uncertainty: the mathematics

This page writes out the pipeline behind `aj fit --uq ard` (revision 2): how
the per-atom force uncertainty is built, calibrated and served. For how to use
it, see [Per-atom force uncertainty](../howto/force-uncertainty.md).

**In one paragraph.** The served force uncertainty is a **shape** times a
per-group **scale**. The shape is the per-atom $3\times3$ block $V(x)$ of an
exact, centred, delete-one-cluster jackknife covariance of the fitted
coefficients, with clusters given by spatial blocks within large cells; its
trace $v(x)=\operatorname{tr}V(x)$ is the isotropic shape. The scale comes in
two forms: `forces_std` uses a per-group rms factor $\lambda_g^{\mathrm{rms}}$,
and `forces_q` a per-group conformal quantile $q_g$ computed with each
configuration weighted equally. Both are fitted on scores whose numerator
(the error) and denominator (the shape) come from the same hold-out
posterior, then carried to the served posterior by a fitted power of the
training-set size. Within a group, the ordering of atoms by $\sigma$ is the
ordering given by the shape; across groups it changes only through ratios of
the group scales.

**Notation.** Atoms $i$; Cartesian components $\alpha\in\{x,y,z\}$; training
configurations $c\in T$; jackknife (sandwich) clusters $k\in\mathcal{K}$;
Mondrian groups $g$; $L$ basis functions. $\mathsf{T}$ is the transpose.

## 1. Model and data

Each atom $i$ with environment $x_i$ has a site descriptor
$\phi(x_i)\in\mathbb{R}^L$: the ACE product basis in species-major blocks,
plus pair blocks. The model is linear in the coefficients $c$:

$$
E(R) = \sum_i \phi(x_i)\cdot c + \sum_i E^0_{Z_i}.
$$

A training configuration contributes energy, force and virial observations,
each linear in $c$:

$$
\phi_E = \sum_i \phi(x_i),\qquad
\phi_{F,i\alpha} = -\frac{\partial \phi_E}{\partial r_{i\alpha}},\qquad
\phi_{V,\alpha\beta} = -\sum_i r_{i\alpha}\frac{\partial\phi_E}{\partial r_{i\beta}}.
$$

Each row has a structural weight $w$ (per-atom energy normalisation and
per-type weights) and a noise scale $\sigma_q$ for its quantity
$q\in\{E,F,V\}$. The whitened rows and targets are
$\psi=\phi\,w/\sigma_q$ and $\tilde y = y\,w/\sigma_q$, where $y$ is the label
minus the $E^0$ baseline.

## 2. Prior and posterior

The prior is $c\sim\mathcal{N}(0,\Lambda^{-1})$ with
$\Lambda=\operatorname{diag}\big(\Gamma_j^2 e^{a_{k(j)}}\big)$. Here
$\Gamma_j$ is the smoothness prior of column $j$, and $a_k$ is the
**automatic relevance determination (ARD)** log-precision of body order
$k(j)\in\{2,3,4\}$; pair columns and correlation-order-1 columns both count as
2-body. The posterior is Gaussian:

$$
A=\sum_{\text{rows}}\psi\psi^\mathsf{T}+\Lambda,\qquad
\bar c = A^{-1}\sum_{\text{rows}}\psi\tilde y,\qquad
\operatorname{Cov}(c)=A^{-1}.
$$

The computation works in the prior-scaled system
$S=D^{-1}AD^{-1}=LL^\mathsf{T}$ with $D=\operatorname{diag}\Gamma$ (here $L$
is the Cholesky factor), and floors $a_k\ge a_{\mathrm{floor}}$ so that
$\operatorname{cond}(S)\le10^{14}$. The served `model.npz` holds $\bar c$.

## 3. Hyperparameters: the evidence

The hyperparameters are
$h=(\log\sigma_E,\log\sigma_F,\log\sigma_V,a_2,a_3,a_4)$. The evidence uses
the per-quantity statistics $G_q=\sum\psi\psi^\mathsf{T}$, $b_q$,
$y^\mathsf{T}y_q$ and $n_q$, and up to constants is

$$
\log p(\mathcal D\mid h) = -\tfrac12\sum_q\frac{y^\mathsf{T} y_q}{\sigma_q^2}
+\tfrac12 b^\mathsf{T} A^{-1}b-\tfrac12\log|A|+\tfrac12\log|\Lambda|-\sum_q n_q\log\sigma_q .
$$

The **joint** mode (`--ard-mode joint`, the default) maximises it over all six
components of $h$, with L-BFGS-B on the gradient-normalised objective. The
**sequential** mode fixes $\sigma_q$ at the linear MAP and fits only the
$a_k$, which needs a single Gram matrix and less memory.

## 4. The hold-out protocol

1. **Stratified split by configuration.** Compute the Mondrian group of every
   training atom ([section 9](#9-the-groups)), with the band edges computed
   from all of $T$. Give each configuration $c$ the stratum $\varsigma(c)$,
   the *most extreme* group it populates: the highest distortion band, with
   ties broken in favour of $z\ne z^\star$. Within each stratum, randomly
   assign a fraction $f$ (`--ard-val-frac`, default 0.2) of configurations to
   $T_{\mathrm{val}}$ and the rest to $T_{\mathrm{fit}}$, rounding so that
   every stratum with at least two configurations has at least one on each
   side. Strata with fewer than two configurations are merged into the next
   less extreme stratum. A crack cell is therefore split as a crack cell, even
   though most of its atoms are bulk-like.
2. **Hold-out posterior.** Fit $h$ on $T_{\mathrm{fit}}$, giving
   $P_{\mathrm{fit}}$ with mean $\bar c_{\mathrm{fit}}$, factor
   $L_{\mathrm{fit}}$ and shape factor $\tilde Q_{\mathrm{fit}}$
   ([section 6](#6-the-shape-exact-centred-jackknife-covariance), built over
   the clusters of $T_{\mathrm{fit}}$ only). For each held-out atom
   $i\in T_{\mathrm{val}}$, record the error
   $e_i=F_i-\hat F_i(\bar c_{\mathrm{fit}})$ and the hold-out shape
   $V_{\mathrm{fit}}(x_i)$. Both are out of sample, because no row of the
   atom's configuration is in $T_{\mathrm{fit}}$.
3. **Served posterior.** Refit on all of $T$, starting from
   $h_{\mathrm{fit}}$. This gives $P$ with mean $\bar c$, factor $L$ and shape
   factor $\tilde Q$.

$P_{\mathrm{fit}}$ is needed only to produce the scores; $\tilde
Q_{\mathrm{fit}}$ is not stored.

## 5. Sandwich clusters

The jackknife treats clusters of rows as independent units. The clusters
$\mathcal{K}$ partition the rows of $T$:

- **Small configurations** are a single cluster containing all their energy,
  force and virial rows. A configuration counts as small unless it can hold at
  least two blocks of side $\ell$ along some lattice direction.
- **Large configurations** are split into one cluster per spatial block,
  containing the force rows of the atoms in that block, and one cluster
  containing the energy row and the six virial rows, since these couple every
  atom in the cell.

Blocks are the cells of a grid with
$n_a=\max\big(1,\lfloor w_a/\ell\rfloor\big)$ divisions along lattice direction
$a$. Here $w_a$ is the cell width perpendicular to the other two lattice
vectors, or the extent of the atomic bounding box along a non-periodic
direction. Atoms are assigned by their fractional coordinates, respecting
periodicity.

**Block size.** A force depends on environments within $r_{\mathrm{cut}}$ of
neighbours within $r_{\mathrm{cut}}$, so the smallest sensible size is
$\ell=2r_{\mathrm{cut}}$. The default is $\ell=3r_{\mathrm{cut}}$
(`--ard-cluster-size 3`; `inf` keeps whole configurations). The size is
checked by a plateau test: increase $\ell$ until $v(x)$ and the
$\lambda_g$ stop changing, standard practice for cluster-robust variances
(Cameron and Miller 2015). On the validation with crack cells in training,
$\ell/r_{\mathrm{cut}}\in\{2,3,4,6,\infty\}$ changed tip coverage by at most
0.005. Sub-clustering acts only on the jackknife, which is built from $T$:
large cells that appear only in a calibration set never enter it.

**Two kinds of unit.** Jackknife clusters are an ingredient of the shape
only. The hold-out split, the conformal quantiles and the weights all
operate on whole configurations.

## 6. The shape: exact centred jackknife covariance

### Leverage-corrected cluster scores

For cluster $k$ with stacked whitened rows
$\Psi_k\in\mathbb{R}^{n_k\times L}$ and residuals at the served mean
$\rho_k=\tilde y_k-\Psi_k\bar c$, define the cluster hat block and the
leverage-corrected (PRESS) score

$$
H_{kk}=\Psi_kA^{-1}\Psi_k^\mathsf{T} = W_k^\mathsf{T} W_k,\quad
W_k=L^{-1}D^{-1}\Psi_k^\mathsf{T},\qquad
\tilde g_k=\Psi_k^\mathsf{T}(I-H_{kk})^{-1}\rho_k .
$$

With $h$ held fixed, deleting the rows of cluster $k$ changes the posterior
mean by exactly

$$
\bar c-\bar c_{(-k)} = A^{-1}\tilde g_k ,
$$

the cluster DFBETA (Cook and Weisberg 1982). Here $(I-H_{kk})^{-1}\rho_k$ are
the residuals that cluster $k$ would have if it were left out of the fit.
The identity is checked to machine precision in the test suite, for a
whole-configuration cluster, a spatial block and an energy/virial cluster,
against an explicit refit.

**Well-posedness.** Since $\Lambda\succ0$, $A\succ\Psi_k^\mathsf{T}\Psi_k$, so
every eigenvalue of $H_{kk}$ lies in $[0,1)$ and $I-H_{kk}$ is always
invertible. Unlike unregularised least squares, no single cluster can
saturate the fit.

**Computation.** If $n_k\le L$, form $W_k$ and solve the $n_k\times n_k$
system above. If $n_k>L$, use the push-through identity

$$
A^{-1}\tilde g_k = A_{(-k)}^{-1}\Psi_k^\mathsf{T}\rho_k,\qquad
A_{(-k)}=A-\Psi_k^\mathsf{T}\Psi_k ,
$$

with a rank-$n_k$ Cholesky downdate of $S$. Clusters are independent and are
batched. Sub-clustering keeps most $n_k$ to $3N_k\lesssim10^3$.

**Block approximation** (`--ard-press block`). Apply the correction per row
block instead: $(1-h)^{-1}$ for the energy row, the $3\times3$ force block of
each atom and the $6\times6$ virial block. This is HC3-like rather than an
exact deletion, and it ignores the coupling between atoms within a cluster.
It is never the default.

### Centred meat and the shape

Let $\bar g=|\mathcal{K}|^{-1}\sum_k\tilde g_k$. The centred "meat" of the
sandwich is

$$
\tilde M=\sum_{k\in\mathcal{K}}(\tilde g_k-\bar g)(\tilde g_k-\bar g)^\mathsf{T} .
$$

Centring removes the mean deletion effect. Without it, the meat contains a
term from the prior's gradient: for the uncorrected scores,
$\sum_k\Psi_k^\mathsf{T}\rho_k=\Lambda\bar c$ at the MAP. With centring,
$A^{-1}\tilde M A^{-1}$ is the delete-one-cluster jackknife covariance of
$\bar c$, up to the factor $(|\mathcal{K}|-1)/|\mathcal{K}|$, which the scale
absorbs. It is the CR3 cluster-robust estimator (MacKinnon, Nielsen and Webb
2023), the exact counterpart of the infinitesimal jackknife. It is a
Huber–White sandwich (Huber 1967; White 1982; Müller 2013): the posterior
covariance $\kappa^2A^{-1}$ is its homoscedastic special case
$\tilde M\propto A$. It is robust to misspecification, which dominates the
error of an ACE fit to noise-free labels.

Define

$$
\tilde Q=S^{-1}D^{-1}\big[\tilde g_1-\bar g\;\cdots\;\tilde g_{|\mathcal{K}|}-\bar g\big]\in\mathbb{R}^{L\times|\mathcal{K}|},
\qquad u_\alpha(x)=D^{-1}\phi_\alpha(x)^\mathsf{T} ,
$$

where $\phi_\alpha(x)$ are the raw, unweighted force rows of a target atom.
The served shape is

$$
V(x)_{\alpha\beta}=u_\alpha^\mathsf{T}\tilde Q\tilde Q^\mathsf{T} u_\beta,\qquad
v(x)=\operatorname{tr}V(x)=\sum_\alpha\big\|\tilde Q^\mathsf{T} u_\alpha\big\|^2 .
$$

$V(x)$ is the full anisotropic block, and costs nothing extra because the
projections $\tilde Q^\mathsf{T} u_\alpha$ are already needed for $v$. Since
$\phi_\alpha A^{-1}(\bar c-\bar c_{(-k)})$ is the change in the prediction at
$x$ when cluster $k$ is deleted, $v(x)$ is the spread of the prediction over
cluster deletions. It is large where the prediction leans on clusters the
model fits badly.

**Epistemic variant** (`--ard-variance kappa`):
$V_\kappa(x)_{\alpha\beta}=\phi_\alpha A^{-1}\phi_\beta^\mathsf{T}$, with
$v_\kappa=\operatorname{tr}V_\kappa$. Everything below applies to either
variant.

### Storage

The cluster count $|\mathcal{K}|$ can exceed $L$ once large cells are
sub-clustered. Since only $\tilde Q\tilde Q^\mathsf{T}$ enters $V$, the
posterior stores a factor $R$ with $RR^\mathsf{T}=\tilde Q\tilde Q^\mathsf{T}$:

- $R=\tilde Q$ if $|\mathcal{K}|\le L$;
- otherwise $R=U_r\Sigma_r$, from the thin SVD
  $\tilde Q=U\Sigma V^\mathsf{T}$ (so that
  $\tilde Q\tilde Q^\mathsf{T}=U\Sigma^2U^\mathsf{T}$), keeping the leading
  $r$ singular triplets.

An optional truncation keeps the smallest $r$ with
$\sum_{j\le r}\sigma_j^2\ge\tau\sum_j\sigma_j^2$ (default $\tau=1$, exact).
Storage is $L\times\min(|\mathcal{K}|,L)$.

### Rotation equivariance

The ACE site descriptor is invariant under a global orthogonal transformation
$O\in O(3)$ (rotation or reflection) of positions and cell, so its position
derivatives, the force rows, transform as vectors:
$\phi_\alpha(Ox)=\sum_\beta O_{\alpha\beta}\,\phi_\beta(x)$. $R$ and $D$ act in
coefficient space and do not see $O$. Hence $u(Ox)=O\,u(x)$ and

$$
V(Ox)=O\,V(x)\,O^\mathsf{T},\qquad \mathtt{forces\_cov}(Ox)=O\,\mathtt{forces\_cov}(x)\,O^\mathsf{T},
$$

while $v=\operatorname{tr}V$, the eigenvalues of $V$, and the scores of
[section 7](#7-calibration-scores) are invariant: for an error that rotates
with the structure, $e\mapsto Oe$,

$$
(Oe)^\mathsf{T}\big(OVO^\mathsf{T}+\epsilon I\big)^{-1}(Oe)
=e^\mathsf{T}\big(V+\epsilon I\big)^{-1}e ,
$$

since $\epsilon$ depends on $\operatorname{tr}V$ only. The groups and the
support features are built from distances and invariant descriptors, so
`forces_std`, `forces_q`, `forces_q_mahal`, `forces_group` and the support
flag are invariant, and every served quantity is permutation-equivariant.
The test suite checks all of this for random rotations and reflections.

## 7. Calibration scores

For $i\in T_{\mathrm{val}}$, both the error and the shape come from
$P_{\mathrm{fit}}$:

$$
\text{isotropic:}\quad s_i=\frac{|e_i|}{\sqrt{v_{\mathrm{fit}}(x_i)/3}},
\qquad
\text{anisotropic:}\quad s_i=\sqrt{e_i^\mathsf{T}\big(V_{\mathrm{fit}}(x_i)+\epsilon_i I\big)^{-1}e_i},
\quad \epsilon_i=\epsilon\,\tfrac13\operatorname{tr}V_{\mathrm{fit}}(x_i).
$$

The default regulariser is $\epsilon=10^{-3}$; it matters only where
$V_{\mathrm{fit}}$ is close to rank-deficient. Under the working model
$e\sim\mathcal{N}(0,\lambda^2\,\tfrac{v}{3}I_3)$ (isotropic) or
$e\sim\mathcal{N}(0,\lambda^2V)$ (anisotropic), $s/\lambda\sim\chi_3$. The
conformal step does not rely on this model; only the reading of `forces_std`
as a Gaussian does. The mode is chosen at fit time
(`--force-shape iso|aniso`, default `aniso`: on the validation it is the only
variant without target-regime calibration that meets the coverage targets),
and every scale below is computed in that mode.

### The transfer assumption

The scores come from a model trained on about $(1-f)|T|$ configurations, but
the scale is applied to the shape of the model trained on $T$. The shape is a
parameter variance, scaling roughly as $1/N$, whereas the error is dominated
by approximation error (the squared rms scale is about 39 on the validation
benchmark). The ratio of squared error to shape is therefore unchanged only
if the error is variance-dominated. If it is bias-dominated, $\lambda^2$
grows roughly in proportion to $N$, and the hold-out scale underestimates the
served one by up to $(1-f)^{-1/2}$. Writing $\lambda\propto N^{\beta}$, the
variance-dominated limit is $\beta=0$ and the bias-dominated one
$\beta=\tfrac12$.

### The transfer exponent

The served scale replaces the uncorrected hold-out scale by an extrapolation
in $N$ estimated in every fit (`--ard-transfer exponent`, the default).
$T_{\mathrm{fit}}$ is split again by the same stratified rule into
$T_{\mathrm{fit2}}$, with $N_{\mathrm{fit2}}\approx(1-f)N_{\mathrm{fit}}$
configurations. A second hold-out posterior $P_{\mathrm{fit2}}$ is fitted on
it, with its own jackknife shape, and scores the same $T_{\mathrm{val}}$
atoms. Let $\lambda_1$ and $\lambda_2$ be the configuration-weighted rms
scales of the $P_{\mathrm{fit}}$ and $P_{\mathrm{fit2}}$ scores, pooled over
all groups with the weights of [section 8](#8-the-scales), over the
$T_{\mathrm{val}}$ atoms at which both shapes are positive. Then

$$
\hat\beta=\frac{\log(\lambda_1/\lambda_2)}{\log(N_{\mathrm{fit}}/N_{\mathrm{fit2}})},\qquad
\beta=\min\bigl(\max(\hat\beta,0),\tfrac12\bigr),\qquad
t=(N/N_{\mathrm{fit}})^{\beta},
$$

and every calibration score is multiplied by $t$ before the per-group scales
are formed.

- The clip keeps $\beta$ between the two limits above, and a clip is logged
  as a warning. If $\hat\beta$ is undefined ($N_{\mathrm{fit2}}=N_{\mathrm{fit}}$,
  or a non-finite scale), $\beta=\tfrac12$, the conservative limit.
- One pooled $\beta$ serves all groups. `--ard-transfer sqrt` fixes
  $\beta=\tfrac12$ and `none` fixes $\beta=0$, without the second fit.
- **Precision.** $\beta$ is a single-split estimate. Because the extrapolation
  step equals the baseline ($N/N_{\mathrm{fit}}\approx N_{\mathrm{fit}}/N_{\mathrm{fit2}}$),
  the factor's relative error is about that of $\lambda_1/\lambda_2$, and the
  clip bounds the factor to at most $(1-f)^{-1/2}$.
- On the validation benchmark, a sweep of $f$ gives
  $\log\lambda\propto0.37\log N_{\mathrm{fit}}$, and
  $(3680/2944)^{0.37}=1.086$ matches the served rms-$z$ deficit of $1.087$;
  the per-fit estimates were $\beta=0.36$–$0.38$ at $f\in\{0.1,0.2,0.3\}$.
- The cost is one more evidence fit and jackknife shape, on about
  $(1-f)^2|T|$ configurations.

`ard.json` records $\lambda_1$, $\lambda_2$, $\hat\beta$, $\beta$ and $t$ under
`"transfer"`.

## 8. The scales

### Units and weights

For group $g$, let $\mathcal{C}_g$ be the set of calibration configurations
with at least one atom in $g$, $n_{\mathrm{cfg},g}=|\mathcal{C}_g|$, and
$n_{c,g}$ the number of atoms of $c$ in $g$. Each calibration atom
$i\in c\cap g$ carries the weight $1/n_{c,g}$, so every configuration has
total weight 1 in every group it populates. A crack cell is one unit in the
tip group, whether it has 3 tip atoms or 30.

### `forces_std`: the per-group rms scale

$$
(\lambda_g^{\mathrm{rms}})^2=\frac{1}{3\,n_{\mathrm{cfg},g}}\sum_{c\in\mathcal{C}_g}\frac{1}{n_{c,g}}\sum_{i\in c\cap g}s_i^2,
$$

$$
\mathtt{forces\_std}(x)=\lambda^{\mathrm{rms}}_{g(x)}\sqrt{v(x)},\qquad
\mathtt{forces\_cov}(x)=\big(\lambda^{\mathrm{rms}}_{g(x)}\big)^2 V(x).
$$

This is the configuration-weighted Gaussian maximum-likelihood scale within
each group. It does not depend on $\alpha$ and makes no coverage claim; it is
the scale to use when propagating uncertainty or when a Gaussian is wanted.
`forces_cov` is served in both modes. In isotropic mode it uses the isotropic
$\lambda_g^{\mathrm{rms}}$, so its trace, $\mathtt{forces\_std}^2$, is
calibrated but its orientation is the uncalibrated shape of $V(x)$.

### `forces_q`: the per-group conformal quantile

The configuration-weighted (pooled-CDF) score distribution of group $g$,
with a test point at $+\infty$, is

$$
\hat F_g(t)=\frac{1}{n_{\mathrm{cfg},g}+1}\Bigg[\sum_{c\in\mathcal{C}_g}\frac{1}{n_{c,g}}\sum_{i\in c\cap g}
\mathbf{1}\{s_i\le t\}\;+\;\mathbf{1}\{+\infty\le t\}\Bigg],
\qquad q_g=\inf\{t:\hat F_g(t)\ge1-\alpha\},
$$

with $1-\alpha$ = `--ard-coverage`. The served regions are

$$
\text{isotropic:}\quad \{\Delta F: |\Delta F|\le \mathtt{forces\_q}(x)\},\qquad
\mathtt{forces\_q}(x)=q_{g(x)}\sqrt{v(x)/3},
$$

$$
\text{anisotropic:}\quad \big\{\Delta F:\Delta F^\mathsf{T}\big(V(x)+\epsilon(x)I\big)^{-1}\Delta F\le q_{g(x)}^2\big\},
$$

with $\epsilon(x)=\epsilon\,\tfrac13\operatorname{tr}V(x)$ as for the scores.
In anisotropic mode, $q_{g(x)}$ is served per atom (`forces_q_mahal`)
together with $V(x)$, and the scalar `forces_q` is the largest semi-axis,
$q_{g(x)}\sqrt{\lambda_{\max}(V+\epsilon I)}$, for convenience.

**The all-groups fallback.** When no neighbouring group qualifies (below),
the pool is all atoms with the same weights $w_i=1/n_{c,g}$, so a
configuration spanning $k$ groups has total weight $k$. The $+\infty$ test
point then carries the weight $W/n_{\mathrm{cfg}}$ of one configuration's
mean total weight, $W=\sum_i w_i$:

$$
\hat F(t)=\frac{\sum_i w_i\mathbf{1}\{s_i\le t\}+(W/n_{\mathrm{cfg}})\,\mathbf{1}\{+\infty\le t\}}{W(1+1/n_{\mathrm{cfg}})} ,
$$

which reduces to the per-group formula for a single group
($W=n_{\mathrm{cfg}}$). In every pool, $q$ is finite if and only if
$n_{\mathrm{cfg}}\ge\lceil(1-\alpha)/\alpha\rceil$ (9 at $1-\alpha=0.9$, 99
at 0.99). If even the all-groups pool is smaller, $q=\infty$ and the fit logs
a warning.

**Coverage statement.** Assume calibration and target configurations are
exchangeable, and take the target to be a uniformly random group-$g$ atom of a
new configuration. Then the region has coverage $\approx1-\alpha$, with an
approximation error controlled by $n_{\mathrm{cfg},g}$, not by the atom count.
This is the pooled-CDF construction of Dunn, Wasserman and Ramdas (2023).
Their exact finite-sample constructions (double conformal, subsampling) are
not implemented. No Gaussianity is assumed.

**Merging.** Groups with $n_{\mathrm{cfg},g}<n_{\min}$ (`--ard-n-min`,
default 20, raised to $\lceil(1-\alpha)/\alpha\rceil$ if that is larger) take
the scales of the nearest qualifying distortion band with the same
$[z=z^\star]$ flag, then across the flag, then of the all-groups pool. The
merges are reported.

**Reported per group.** Each atom is served its group id, `forces_group`. The
per-group table stored with the posterior holds $n_{\mathrm{cfg},g}$ (split
into configurations from $T_{\mathrm{val}}$ and from calibration sets $U$),
total atoms, $\lambda_g^{\mathrm{rms}}$, $q_g$, the merges, and the
Gaussianity diagnostic

$$
r_g=\frac{q_g}{\lambda_g^{\mathrm{rms}}\,\chi^{-1}_3(1-\alpha)} .
$$

$r_g\approx1$ means that reading `forces_std` as a Gaussian reproduces the
conformal coverage in that group. $r_g>1$ means the score distribution has
heavier tails than $\chi_3$; within the group, `forces_q` is then $r_g$ times
the radius implied by `forces_std` read as a Gaussian. $r_g$ is constant
within a group, so it is not served per atom: `forces_group` joins each atom
to it.

**What is invariant.** The mean $\bar c$; the energy and virial predictions;
the shape; and the ordering by $\sigma$ within a group. Across groups, the
relative $\sigma$ changes by
$\lambda^{\mathrm{rms}}_g/\lambda^{\mathrm{rms}}_{g'}$ and the relative
radius by $q_g/q_{g'}$.

## 9. The groups

Three sets of constants are fixed at fit time and stored with the posterior:
$r_1$, the first minimum of the training radial distribution function;
$z^\star$, the modal training coordination; and the band edges. For each
atom,

$$
z_i=\#\{j:r_{ij}<r_1\},\qquad
d_i=\frac{\operatorname{std}\{r_{ij}:r_{ij}<r_1\}}{\operatorname{mean}\{r_{ij}:r_{ij}<r_1\}}
\quad(\text{undefined if } z_i<2).
$$

The band edges are the 50th, 90th and 99th percentiles of $d$ over all atoms
of $T$, and an atom with undefined $d$ goes in the top band. The group is
$g=\operatorname{band}(d)\times[z\ne z^\star]$, numbered
$2\,\operatorname{band}(d)+[z\ne z^\star]$, so $G=8$ before merging
(`--ard-groups none` drops the bands, $G=2$; tied percentiles collapse
bands). The same edges stratify the hold-out split. They depend only on
geometry, never on labels or scores, so letting $T_{\mathrm{val}}$'s geometry
into three quantiles over thousands of atoms has a negligible effect on
exchangeability. What matters is that the edges are frozen at fit time:
`aj calibrate` never changes which group an atom is in.

## 10. Calibration sets: `aj calibrate`

**Inside the fit**, the pool is $T_{\mathrm{val}}$, with the scores of
[section 7](#7-calibration-scores).

**`aj calibrate`** takes a set $U$ of labelled configurations from the target
regime, disjoint from $T$. The errors are those of the served model,
$e_u=F_u-\hat F_u(\bar c)$, and the shape is the served $V(x_u)$, which is
out of sample because $U\cap T=\emptyset$. The scores follow section 7 with
$V$ in place of $V_{\mathrm{fit}}$ (and no transfer factor, since they already
come from the served model). Then, for each group $g$:

- if $U$ has at least $n_{\min}$ configurations in $g$, the pool for $g$ is
  $U$ alone (**per-group replace**, the default);
- otherwise the pool is $T_{\mathrm{val}}\cup U$, with configuration weights
  as in [section 8](#8-the-scales).

`--append` forces $T_{\mathrm{val}}\cup U$ in every group, and `--replace`
forces $U$ alone in every group (merging as needed).
$\lambda_g^{\mathrm{rms}}$ and $q_g$ are recomputed, and each group reports
its pool composition. Mixed groups are where the transfer assumption matters
most, since $T_{\mathrm{val}}$ scores come from $P_{\mathrm{fit}}$ and $U$
scores from the served model.

**Rationale.** A target atom resembling $U$ should be calibrated against $U$.
Mixing in $T_{\mathrm{val}}$ scores from easier atoms of the same group
dilutes the quantile and gives the coverage of a mixture. The other side of
this is that a posterior calibrated on $U$ is specific to the regime of $U$
(see the [validation](../howto/force-uncertainty.md#validation)).

## 11. The support diagnostic

1. **Features.** For each species, fit a whitened PCA of the site descriptor
   $\phi(x)$ (that species' block) on the training atoms, keeping components
   up to 99 % explained variance, at most 64. The site descriptor is
   rotation-invariant and includes all body orders. The force-row projections
   $\tilde Q^\mathsf{T}u_\alpha$ are not used as features, because they rotate
   with the structure.
2. **Density ratio.** For each species, fit an L2-regularised logistic
   classifier (target = 1, calibration = 0, class-balanced) on these
   features, with the L2 strength chosen by 5-fold cross-validation grouped by
   configuration. This gives
   $w(x)\propto P(\mathrm{target}\mid x)/P(\mathrm{cal}\mid x)$.
3. **Weighted quantile** (Tibshirani, Barber, Candès and Ramdas 2019).
   Calibration atom $i$ has mass $w(x_i)/n_{c(i)}$, which combines the shift
   weight with the configuration weight; the test atom has mass $w(x)$. With
   $W$ the total mass,

    $$
    q_w(x)=Q_{1-\alpha}\Big(\sum_{i\in\mathrm{cal}}\frac{w(x_i)}{n_{c(i)}W}\,\delta_{s_i}+\frac{w(x)}{W}\,\delta_{+\infty}\Big).
    $$

4. **Report** `support_q` $=q_w(x)$ and `support_ok` $=[q_w<\infty]$ per
   atom, and $n_{\mathrm{eff}}=(\sum_i p_i)^2/\sum_i p_i^2$ per species, over
   the calibration masses $p_i=w(x_i)/n_{c(i)}$.

**Caveats.** Richer features give more extreme ratios and so a smaller
$n_{\mathrm{eff}}$; this is the honest outcome. The diagnostic assumes that
the score distribution given the classifier features is unchanged from
calibration to target. The labels are deterministic in the full structure but
not in the local environment ([limitations](#13-assumptions-and-limitations)),
so this is an approximation even with full features. The diagnostic is not
used to scale $\sigma$. `aj calibrate` updates the reference with the same
replace or append rule as the scores.

## 12. End to end

fit
:   data → group features (fixing $r_1$, $z^\star$ and the band edges from
    $T$) → stratified split → evidence fit on $T_{\mathrm{fit}}$ →
    $P_{\mathrm{fit}}$ → clusters, PRESS scores, centred
    $\tilde Q_{\mathrm{fit}}$ → $V_{\mathrm{fit}}(x_i)$ and errors $e_i$ on
    $T_{\mathrm{val}}$ → scores $s_i$ → (second split, $P_{\mathrm{fit2}}$,
    transfer factor $t$) → refit on $T$ → $P$, clusters, PRESS scores,
    centred $\tilde Q$ → stored factor $R$ → per-group
    $\lambda^{\mathrm{rms}}_g$ and $q_g$ from $t\,s_i$ (configuration-weighted,
    merged below $n_{\min}$).

calibrate
:   labelled target set $U$ → scores with the served model and shape →
    per-group replace (or append, or replace) → recompute
    $\lambda^{\mathrm{rms}}_g$ and $q_g$.

serve
:   shape $V(x)$ from the projections $R^\mathsf{T}u_\alpha(x)$ →
    `forces_std`, `forces_cov`, `forces_q` (and `forces_q_mahal` in
    anisotropic mode) and `forces_group`.

diagnose
:   unlabelled target → descriptor PCA → $w(x)$ → weighted quantile →
    `support_ok`, `support_q`, $n_{\mathrm{eff}}$.

## 13. Assumptions and limitations

1. **Exchangeability at configuration level, within a group.** The coverage
   statement is approximate, with error controlled by $n_{\mathrm{cfg},g}$,
   which is reported with the merges.
2. **Transfer** from the hold-out model to the served model. This is an
   empirical assumption: the per-fit exponent extrapolates the scale from two
   hold-out sizes to $|T|$, assuming a power law in $N$ with exponent in
   $[0,\tfrac12]$. On out-of-distribution cells some dependence on $f$
   remains (crack-tip coverage 0.896 at $f=0.1$, 0.851 at $f=0.3$), although
   the in-distribution scale is independent of $f$.
3. **The shape is a parameter variance used as a proxy** for where the
   approximation error is large. $v(x)$ measures how much the prediction at
   $x$ depends on which clusters were in the fit, not the approximation error
   itself; its rank correlation with the error is 0.3–0.4.
4. **Independence of jackknife clusters.** Spatial blocks assume that errors
   are uncorrelated across block boundaries beyond about $2r_{\mathrm{cut}}$.
   The block-size plateau tests this.
5. **Hyperparameters fixed in the jackknife.** The deletion jackknife holds
   $h$ fixed, so the shape carries no uncertainty from the hyperparameters.
6. **Groups chosen from benchmark evidence.** A shift the groups do not
   resolve (chemical short-range order, a new phase) is covered only
   marginally. The support diagnostic is the guard.
7. **Forces only.** Energy and virial uncertainties are not calibrated.
8. **Locality acts as effective noise.** The labels carry no noise, but the
   descriptor truncates the environment at $r_{\mathrm{cut}}$. Conditional on
   $x$, the label is therefore not deterministic: non-local elastic and
   electronic contributions act as noise relative to the model. The scale
   absorbs its size; nothing models where it is larger. This is a candidate
   explanation for the remaining gap at crack tips.

## References

- P. J. Huber, The behavior of maximum likelihood estimates under nonstandard
  conditions, *Proc. 5th Berkeley Symp.* (1967); H. White, *Econometrica*
  **50**, 1 (1982).
- U. K. Müller, Risk of Bayesian inference in misspecified models, and the
  sandwich covariance matrix, *Econometrica* **81**, 1805 (2013).
- R. D. Cook and S. Weisberg, *Residuals and Influence in Regression*,
  Chapman & Hall (1982).
- J. G. MacKinnon, M. Ø. Nielsen and M. D. Webb, Cluster-robust inference: a
  guide to empirical practice, *J. Econometrics* **232**, 272 (2023).
- A. C. Cameron and D. L. Miller, A practitioner's guide to cluster-robust
  inference, *J. Human Resources* **50**, 317 (2015).
- R. Dunn, L. Wasserman and A. Ramdas, Distribution-free prediction sets for
  two-layer hierarchical models, *J. Am. Stat. Assoc.* (2023),
  [arXiv:1809.07441](https://arxiv.org/abs/1809.07441).
- R. J. Tibshirani, R. F. Barber, E. J. Candès and A. Ramdas, Conformal
  prediction under covariate shift, *NeurIPS* (2019),
  [arXiv:1904.06019](https://arxiv.org/abs/1904.06019).
