"""Matrix products at full precision.

On Ampere and later GPUs XLA's default for float32 dot products is TF32 (a 10-bit mantissa in the
multiplies).  For these models that is not a small loss: the on-site blocks of a float32
prediction came out wrong by 0.5 eV and the band energy of a 64-atom cell by a thousand eV
(docs/dev/ham-results.md).  Every traced function of `operators` and `ham` that holds a
contraction is therefore traced under `jax.default_matmul_precision("highest")`: the precision is
recorded on each dot product when it is traced, so the decorator must wrap the function that is
traced, not the call of the compiled one.  float64 is not affected."""
import functools

import jax


def highest(fn):
    """`fn` with its dot products at full precision (see the module docstring)."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with jax.default_matmul_precision("highest"):
            return fn(*args, **kwargs)
    return wrapped
