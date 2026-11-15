"""python kernel_wheel_check.py FIXTURES [VARIANT]: an installed ace-jax platform wheel runs its prebuilt C++
kernel with no compiler (kernel-wheels.yml, in a clean environment outside the checkout).  Not a pytest file.
The prebuilt library is the one `ACECalculator(ffi=True)` and `ffi="auto"` take, and its energy, forces and
stress are the JAX model's to the radial table's tolerance.  VARIANT (e.g. x86-64): that variant, forced."""
import os
import pathlib
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

fixtures = pathlib.Path(sys.argv[1])
if len(sys.argv) > 2:
    os.environ["ACEJAX_FFI_VARIANT"] = sys.argv[2]
import jaxlib  # noqa: E402
from ase.build import bulk  # noqa: E402

import ace_jax  # noqa: E402
from ace_jax import ACECalculator  # noqa: E402
from ace_jax.eval import ffi  # noqa: E402

assert "site-packages" in ace_jax.__file__, f"not an installed wheel: {ace_jax.__file__}"
ffi.compiler = lambda env=None: None                    # a machine without a compiler
pre = ffi.prebuilt()
assert pre is not None, f"no prebuilt kernel in {ffi.PREBUILT} for this machine"
lib, variant = pre
assert len(sys.argv) < 3 or variant == sys.argv[2], variant
assert ffi.library() == lib
worst = 0.0
for name in ["si_fitted", "si_m710", "sige_nofit"]:
    path = str(fixtures / f"{name}.npz")
    at = bulk("Si", "diamond", a=5.43, cubic=True) * (3, 3, 3)
    at.rattle(0.05, seed=1)
    if name == "sige_nofit":
        at.numbers[::2] = 32
    at.calc = ACECalculator(path, ffi=False)
    e0, f0, s0 = at.get_potential_energy(), at.get_forces(), at.get_stress()
    for opt, threads in ((True, 0), (True, 3), ("auto", 1)):
        calc = ACECalculator(path, ffi=opt, ffi_threads=threads)
        at.calc = calc
        e, f, s = at.get_potential_energy(), at.get_forces(), at.get_stress()
        assert calc.ffi and calc.ffi["library"] == str(lib), (calc.ffi, calc.ffi_reason)
        worst = max(worst, abs(e - e0) / abs(e0), np.abs(f - f0).max() / np.abs(f0).max(),
                    np.abs(s - s0).max() / np.abs(s0).max())
assert worst < 1e-6, worst
print(f"ok: prebuilt kernel {variant} ({lib.name}), jaxlib {jaxlib.__version__}, python {sys.version_info[:2]}: "
      f"worst relative difference to the JAX model {worst:.1e}")
