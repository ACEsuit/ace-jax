"""What-if probes of CPU formulations of the ace-jax force call, by patching
model methods in this process only (src/ is unchanged); docs/dev/cpu-gap-profile.md.

    PYTHONPATH=bench:src taskset -c 0 python bench/perf/cpu_probe.py <model> <system> <n> \
        [--aa prod|explicit|fm] [--spline einsum|sum] [--chunk N] [--dtype float64] [--reps 10]

Times the calculator's compiled skin step (`step_s`, as `cpu_gap.py`) and
reports its energy and forces against the unpatched step (max |dE|, max |dF|),
so every variant is checked for parity in the same run.

--aa      the product basis AA = prod_t A[spec_t]:
          prod      as shipped (`jnp.prod(A[:, g], axis=-1)` for ACE, node-major;
                    `jnp.prod(At[s.T], axis=0)` for PACE, feature-major)
          explicit  the same layout, as an explicit chain of multiplies
                    A[:, g0] * A[:, g1] * ... (no prod, so no prod-JVP pad/slice
                    machinery and no (order, n_v, n) gather intermediate)
          fm        (ACE) feature-major: At = A.T, AA_t = At[g0] * At[g1] * ...,
                    rows of n atoms contiguous; the adjoint is whole-row
                    scatter-adds.  (PACE is already feature-major: same as explicit)
--spline  the cubic B-spline of `radial.spline_eval`:
          einsum    as shipped (a batched (1x4)@(4xF) dot per edge)
          sum       w0*c0 + w1*c1 + w2*c2 + w3*c3 elementwise
"""
import argparse
import functools
import json
import operator
import os
import statistics
import sys
import time


def set_chunk(chunk):
    """CHUNK_NODES for this process (the default of energy_forces_virial_dense)."""
    from ace_jax.eval import edge_model
    f = edge_model.EdgeSiteModel.energy_forces_virial_dense
    d = list(f.__defaults__)
    d[0] = chunk
    f.__defaults__ = tuple(d)


def patch(aa, spline):
    import jax
    import jax.numpy as jnp

    from ace_jax.eval import model as acemod
    from ace_jax.eval import pace_model, radial
    if spline == "sum":
        def spline_eval(x, coefs, x0, h, n):
            idx, w = radial._bspline_rows(x, x0, h, n)
            g = coefs[idx]                                          # (..., 4, F)
            return (w[..., 0, None] * g[..., 0, :] + w[..., 1, None] * g[..., 1, :]
                    + w[..., 2, None] * g[..., 2, :] + w[..., 3, None] * g[..., 3, :])
        radial.spline_eval = spline_eval
    mul = lambda xs: functools.reduce(operator.mul, xs)
    if aa == "explicit":
        def _aa(self, A, specs=None):
            specs = self.aa_specs if specs is None else specs
            return jnp.concatenate([mul([A[:, g[:, k]] for k in range(g.shape[1])])
                                    for g in specs], axis=-1)
        acemod.ACEModel._aa = _aa
    elif aa == "fm":
        def _aa(self, A, specs=None):
            specs = self.aa_specs if specs is None else specs
            At = A.T
            return jnp.concatenate([mul([At[g[:, k]] for k in range(g.shape[1])])
                                    for g in specs], axis=0).T
        acemod.ACEModel._aa = _aa
    if aa in ("explicit", "fm"):
        def _node_energies_t(self, At, cr, d, dcin, segment_ids, n_nodes, node_z):
            AA = jnp.concatenate([mul([At[s[:, k]] for k in range(s.shape[1])])
                                  for s in self.aa_specs], axis=0)
            ct = self.ctilde_real().reshape(self.n_aa, -1)
            rho_all = jnp.matmul(ct.T, AA, precision=jax.lax.Precision.HIGHEST)
            rho = rho_all.reshape(self.nz, self.ndensity, n_nodes)[node_z, :, jnp.arange(n_nodes)]
            return self._energy_tail(rho, cr, d, dcin, segment_ids, n_nodes, node_z)
        pace_model.PACEModel._node_energies_t = _node_energies_t


def run(model, system, n, dtype, reps):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ase.calculators.calculator import all_changes

    from ace_jax.calc.point import ACECalculator
    from scaling.structures import supercell
    at = supercell(system, n)
    calc = ACECalculator(model, dtype=getattr(jnp, dtype))
    calc.calculate(at, ["energy", "forces", "stress"], all_changes)
    E, F = calc.results["energy"], calc.results["forces"].copy()
    st = calc._skin_state
    u = jax.device_put(st.displacements(at.positions, np.dtype(dtype)))
    f = calc._step()
    for _ in range(2):
        jax.block_until_ready(f(u, st.arrays, K=st.K))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(f(u, st.arrays, K=st.K))
        ts.append(time.perf_counter() - t0)
    return E, F, statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("system")
    ap.add_argument("n", type=int)
    ap.add_argument("--aa", default="prod")
    ap.add_argument("--spline", default="einsum")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=None, help="variant only: CHUNK_NODES")
    a = ap.parse_args()
    if a.threads == 1:
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false").strip()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_enable_x64", a.dtype == "float64")
    import numpy as np
    E0, F0, t0 = run(a.model, a.system, a.n, a.dtype, a.reps)
    patch(a.aa, a.spline)
    if a.chunk:
        set_chunk(a.chunk)
    jax.clear_caches()
    E1, F1, t1 = run(a.model, a.system, a.n, a.dtype, a.reps)
    print(json.dumps({"model": os.path.basename(a.model), "system": a.system, "n": a.n,
                      "dtype": a.dtype, "aa": a.aa, "spline": a.spline, "chunk": a.chunk,
                      "cpus": sorted(os.sched_getaffinity(0)),
                      "step_s_base": t0, "step_s_variant": t1, "speedup": t0 / t1,
                      "dE": abs(E1 - E0), "dF_max": float(np.abs(F1 - F0).max()),
                      "loadavg": open("/proc/loadavg").read().split()[:3]}))


if __name__ == "__main__":
    sys.exit(main())
