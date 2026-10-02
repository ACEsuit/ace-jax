"""fit.block_lbfgs: curvature-matched block scaling and the joint -> alternating fallback,
on toy objectives with known curvature."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.fit import block_lbfgs as bl


def quad(blocks, a, b, c):
    """0.5 a |x|^2 + 0.5 b |y|^2 + c (x . y[:x.size]) + a small quartic to make it non-trivial."""
    x, y = blocks
    return 0.5 * a * x @ x + 0.5 * b * y @ y + c * x @ y[: x.size] + 1e-3 * jnp.sum(x ** 4)


def _blocks():
    rng = np.random.default_rng(0)
    return (jnp.asarray(rng.standard_normal(4)), jnp.asarray(rng.standard_normal(6)))


def test_pack_unpack_roundtrip():
    blocks = (jnp.arange(6.0).reshape(2, 3), jnp.arange(4.0) + 10)
    shapes = tuple(b.shape for b in blocks)
    r = jnp.asarray([1.0, 7.3])
    for sel in (None, 0, 1):
        xb, other = bl.pack(blocks, r, sel)
        back = bl.unpack(xb, other, r, sel, shapes)
        for b0, b1 in zip(blocks, back):
            np.testing.assert_allclose(np.asarray(b1), np.asarray(b0), rtol=1e-15)


def test_objective_is_invariant_to_the_scaling():
    blocks, args = _blocks(), (1.0, 1e6, 0.1)
    shapes = tuple(b.shape for b in blocks)
    f0 = float(quad(blocks, *args))
    for r in ([1.0, 1.0], [1.0, 1e3], [1.0, 0.1]):
        rj = jnp.asarray(r)
        for sel in (None, 0, 1):
            xb, other = bl.pack(blocks, rj, sel)
            np.testing.assert_allclose(float(bl._packed(xb, other, rj, args, quad, sel, shapes, ())), f0, rtol=1e-13)


@pytest.mark.parametrize("b,expect", [(1e6, 1e3), (4.0, 2.0), (1e-6, bl.R_MIN)])
def test_curvature_scales_match_block_curvature(b, expect):
    """Isotropic uncoupled blocks: h_b is the block's curvature exactly (quartic vanishes at 0)."""
    blocks = (jnp.zeros(4), jnp.zeros(6))
    r, h = bl.curvature_scales(quad, blocks, args=(1.0, b, 0.0))
    np.testing.assert_allclose(h, [1.0, b], rtol=1e-12)
    np.testing.assert_allclose(r, [1.0, expect], rtol=1e-12)          # third case: clamped at R_MIN


def test_tangent_probe_is_live_and_gauge_free():
    X = jnp.asarray(np.random.default_rng(1).standard_normal((3, 5)))
    live = jnp.asarray([[1.0], [0.0], [1.0]])
    v = bl.tangent_probe(X, live, jax.random.PRNGKey(0), row_gauge=True)
    np.testing.assert_allclose(np.asarray(jnp.sum(v * X, -1)), 0.0, atol=1e-12)   # orthogonal to each row
    assert np.all(np.asarray(v[1]) == 0.0)                                           # dead row untouched


def test_joint_mode_converges_on_badly_scaled_problem():
    blocks, args = _blocks(), (1.0, 1e6, 0.5)
    f0 = float(quad(blocks, *args))
    out, info = bl.block_lbfgs(quad, blocks, steps=60, round_steps=20, args=args, tol=0.0)
    f1 = float(quad(out, *args))
    print(f"f0={f0:.3e} f1={f1:.3e} precond={info['precond']} reasons={info['reasons']}")
    assert f1 < 1e-8 * f0
    assert abs(info["precond"][0][1] / 1e3 - 1.0) < 0.2           # ~ sqrt(1e6 / 1)


def test_alternating_mode_splits_each_round():
    blocks, args = _blocks(), (1.0, 4.0, 0.0)
    out, info = bl.block_lbfgs(quad, blocks, steps=10, round_steps=5, mode="alternating", args=args, tol=0.0)
    assert info["precond"] == [[1.0, 1.0], [1.0, 1.0]] and info["curvature"] == []
    assert float(quad(out, *args)) < float(quad(blocks, *args))
    with pytest.raises(ValueError, match="mode"):
        bl.block_lbfgs(quad, blocks, steps=1, mode="nope", args=args)


def test_joint_linesearch_failure_falls_back_to_alternating(monkeypatch):
    real, calls = bl.radial_learn.lbfgs_loop, []

    def fake(f, x0, *, steps, statics, **kw):
        calls.append((statics[1], steps))
        if statics[1] is None:                                    # joint: one accepted step, then fail
            x, fx, trace, _ = real(f, x0, steps=1, statics=statics, **kw)
            return x, fx, trace, "linesearch"
        return real(f, x0, steps=steps, statics=statics, **kw)

    monkeypatch.setattr(bl.radial_learn, "lbfgs_loop", fake)
    _, info = bl.block_lbfgs(quad, _blocks(), steps=12, round_steps=6, args=(1.0, 1e3, 0.1), tol=0.0)
    assert calls == [(None, 6), (0, 2), (1, 2)] * 2                # 6 - 1 accepted - 1 failed = 4 left
    assert info["fallbacks"] == 2 and info["steps"] == 12


def test_hooks_and_single_compile():
    """normalise / after_round run once per round; changing arg VALUES between rounds
    reuses the compiled step."""
    from ace_jax.fit.radial_learn import _lbfgs_step
    seen = []

    def normalise(blocks):
        seen.append("n")
        return blocks

    def after_round(blocks, args):
        seen.append("a")
        a, b, c = args
        return (a * 1.01, b, c)

    kw = dict(round_steps=2, args=(1.0, 1e2, 0.1), normalise=normalise, after_round=after_round, tol=0.0)
    bl.block_lbfgs(quad, _blocks(), steps=2, **kw)                 # warm the cache
    n0 = _lbfgs_step._cache_size()
    seen.clear()
    bl.block_lbfgs(quad, _blocks(), steps=6, **kw)
    assert seen == ["n", "a"] * 3
    assert _lbfgs_step._cache_size() == n0
