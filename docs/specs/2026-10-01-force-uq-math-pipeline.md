# Per-atom force-uncertainty pipeline — mathematics

**Superseded by the LaTeX document `tex/force-uq-math-pipeline.tex` (revision 2, 1 October 2026).**
That document is the binding mathematical reference for `2026-09-30-conformal-force-sigma-design.md`.
Build it with `cd docs/specs/tex && latexmk -pdf force-uq-math-pipeline.tex`.

Revision 2 changes the following:
- the shape is now an exact, centred, delete-one-cluster (PRESS / CR3) jackknife, with spatial
  sub-clustering of large cells and an optional anisotropic 3×3 block;
- the scale is split into a per-group rms factor (`forces_std`, `forces_cov`) and a per-group
  configuration-weighted conformal quantile (`forces_q`);
- calibration scores come from the hold-out posterior alone;
- the split is stratified by group;
- `calibrate` replaces a group's pool by default;
- the support diagnostic classifies on PCA of the full site descriptor.
