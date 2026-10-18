"""Experiment: float32 ace-jax with its matmuls on the A100's TF32 tensor cores (docs/dev/fp32-results.md).

ace-jax traces every float32 contraction at highest precision (`Precision.HIGHEST`, `highest_precision()`): TF32
corrupted float32 force rows in fitting.  On an A100 a true-f32 matmul runs on the FP32 units (19.5 TFLOPS), as
fast as float64 on the FP64 tensor cores, so the float32 bundle gains only 1.1-1.4x there.  `install(preset)`
swaps the algorithm of every float32 x float32 `dot_general` traced afterwards for a `jax.lax.DotAlgorithmPreset`
(TF32_TF32_F32 = TF32; TF32_TF32_F32_X3 = 3-pass TF32, about f32 accuracy; BF16_BF16_F32_X3 / _X6), whatever
precision the call site asked for; float64 dots are left alone, so the float64 reference and the model's float64
preparation (lean, fold, splines) are untouched.  A bench hook, not library API: `dotalg/sitecustomize.py`
installs it in every Python process when $ACEJAX_EXP_DOT_ALGORITHM names a preset (put bench/fp32/dotalg first
on PYTHONPATH), which covers the timing driver's export children.

    ACEJAX_EXP_DOT_ALGORITHM=TF32_TF32_F32 uv run --no-sync python bench/fp32/dot_algorithm.py MODEL.npz [...]

prints `calc.precision.precision_check` (float32 with the preset against float64) on rattled diamond cells.
"""
import os

PRESETS = ("F32_F32_F32", "TF32_TF32_F32_X3", "TF32_TF32_F32", "BF16_BF16_F32_X3", "BF16_BF16_F32_X6")


def install(preset):
    """Every later float32 x float32 dot_general uses `preset` (a DotAlgorithmPreset name)."""
    import jax
    import numpy as np
    from jax._src.lax import lax as L
    algo = getattr(jax.lax.DotAlgorithmPreset, preset)
    orig = L.dot_general
    if getattr(orig, "_acejax_exp", None) == preset:
        return

    def dot_general(lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None, **kw):
        if np.dtype(getattr(lhs, "dtype", None) or np.float64) == np.float32 == np.dtype(
                getattr(rhs, "dtype", None) or np.float64):
            precision, preferred_element_type = algo, None    # the preset fixes the accumulation type
        return orig(lhs, rhs, dimension_numbers, precision, preferred_element_type, **kw)
    dot_general._acejax_exp = preset
    L.dot_general = dot_general


def install_from_env():
    p = os.environ.get("ACEJAX_EXP_DOT_ALGORITHM")
    if p:
        install(p)
    return p


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--cells", nargs="+", type=int, default=[2, 4], help="diamond supercell repeats (8 n^3 atoms)")
    ap.add_argument("--rattle", type=float, default=0.05)
    ap.add_argument("--spline-tol", type=float, default=1e-10)
    a = ap.parse_args()
    preset = install_from_env() or "F32_F32_F32"
    import jax
    jax.config.update("jax_enable_x64", True)
    from ase.build import bulk

    import ace_jax as aj
    from ace_jax.calc.precision import precision_check
    for path in a.models:
        model, meta, _ = aj.load(path)
        for n in a.cells:
            at = bulk("Si", "diamond", a=5.43, cubic=True) * (n, n, n)
            at.rattle(a.rattle, seed=1)
            chk = precision_check(model, meta, at, dtype="float32", lean=True, spline_tol=a.spline_tol)
            print(json.dumps({"model": path, "preset": preset, "device": jax.devices()[0].device_kind, **chk}),
                  flush=True)


if __name__ == "__main__":
    main()
