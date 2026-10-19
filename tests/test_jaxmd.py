"""ace_jax.jaxmd: an ace-jax model as a JAX-MD energy / force function, against ACECalculator."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from ase.build import bulk  # noqa: E402
from conftest import FIXTURE_DIR, require_optional  # noqa: E402

jax_md = require_optional("jax_md")   # the `jaxmd` extra; ACEJAX_REQUIRE_OPTIONAL makes a missing one fail
from jax_md import partition, quantity, simulate, space  # noqa: E402

from ace_jax import jaxmd  # noqa: E402
from ace_jax import load as aj_load  # noqa: E402
from ace_jax.calc.point import ACECalculator  # noqa: E402

MODEL = FIXTURE_DIR / "si_fitted.npz"


@pytest.fixture(scope="module")
def cell():
    at = bulk("Si", "diamond", 5.43, cubic=True) * (3, 3, 3)
    at.positions += np.random.default_rng(0).normal(0, 0.1, at.positions.shape)
    at.set_cell(at.cell * np.array([1.0, 1.02, 0.99]), scale_atoms=True)
    at.calc = ACECalculator(str(MODEL))
    ref = at.get_potential_energy(), at.get_forces(), at.get_stress(voigt=False)
    box = jnp.asarray(at.cell.array.T)
    disp, shift = space.periodic_general(box, fractional_coordinates=True)
    return at, ref, box, disp, shift, jnp.asarray(at.get_scaled_positions())


def _list(disp, box, rc, R, fmt=partition.Dense):
    nf = partition.neighbor_list(disp, box, rc, dr_threshold=1.0, format=fmt, fractional_coordinates=True)
    return nf, jaxmd.allocate(nf, R)


@pytest.mark.parametrize("fmt, compact", [(partition.Dense, False), (partition.Dense, True),
                                          (partition.Sparse, False)])
def test_energy_forces_stress_match_the_calculator(cell, fmt, compact):
    at, (E0, F0, S0), box, disp, _, R = cell
    pot = jaxmd.potential(str(MODEL), disp, at.numbers)
    _, nb = _list(disp, box, pot.cutoff, R, fmt)
    if compact:
        pot = jaxmd.potential(str(MODEL), disp, at.numbers, K=jaxmd.capacity(R, nb, disp, pot.cutoff))
        assert pot.K < nb.idx.shape[1]
    assert abs(float(jax.jit(pot.energy)(R, nb)) - E0) < 1e-10 * len(at)
    F = jax.jit(quantity.force(pot.energy))(R, nb)          # Cartesian under fractional coordinates
    np.testing.assert_allclose(F, F0, atol=1e-10)
    # JAX-MD's stress is minus ASE's
    np.testing.assert_allclose(quantity.stress(pot.energy, R, box, neighbor=nb), -S0, atol=1e-12)
    np.testing.assert_allclose(jax.jit(pot.force)(R, nb), F0, atol=1e-10)   # every format


def test_row_blocks_are_exact_and_sized_as_the_bundles(cell):
    """chunk="auto" (one block at 216 rows) and 64-row blocks under lax.map + checkpoint agree;
    the auto size is the lammps-jax bundles' rule (block_rows_for on the full model)."""
    from ace_jax.export.lammps import BUNDLE_BLOCK_ROWS, MIN_BLOCK_ROWS
    at, _, box, disp, _, R = cell
    pot = jaxmd.potential(str(MODEL), disp, at.numbers)
    _, nb = _list(disp, box, pot.cutoff, R)
    K = jaxmd.capacity(R, nb, disp, pot.cutoff)
    a, b = (jaxmd.potential(str(MODEL), disp, at.numbers, K=K, chunk=c) for c in ("auto", 64))
    assert abs(float(jax.jit(a.energy)(R, nb)) - float(jax.jit(b.energy)(R, nb))) < 1e-12 * len(at)
    np.testing.assert_allclose(jax.jit(a.force)(R, nb), jax.jit(b.force)(R, nb), atol=1e-12)
    m = aj_load(str(MODEL))[0]
    assert jaxmd.block_rows(m, 216, K, 8) == 216
    assert MIN_BLOCK_ROWS <= jaxmd.block_rows(m, 1 << 20, 54, 8) <= BUNDLE_BLOCK_ROWS


def test_overflow_is_nan_not_truncation(cell):
    at, _, box, disp, _, R = cell
    pot = jaxmd.potential(str(MODEL), disp, at.numbers)
    _, nb = _list(disp, box, pot.cutoff, R)
    small = jaxmd.potential(str(MODEL), disp, at.numbers, K=jaxmd.capacity(R, nb, disp, pot.cutoff) - 8)
    assert np.isnan(float(small.energy(R, nb)))
    assert np.isnan(np.asarray(small.force(R, nb))).all()


def test_nve_conserves_energy_and_survives_reallocation(cell):
    at, _, box, disp, shift, R = cell
    pot = jaxmd.potential(str(MODEL), disp, at.numbers)
    nf, nb = _list(disp, box, pot.cutoff, R)
    pot = jaxmd.potential(str(MODEL), disp, at.numbers, K=jaxmd.capacity(R, nb, disp, pot.cutoff + 1.0))
    init, apply = simulate.nve(pot.force, shift, 1.0 / 10.18)           # 1 fs
    st = init(jax.random.PRNGKey(0), R, mass=28.0855, kT=8.617e-5 * 600, neighbor=nb)

    @jax.jit
    def run(st, nb):
        st, nb = jax.lax.fori_loop(0, 50, lambda i, c: (apply(c[0], neighbor=c[1].update(c[0].position)),
                                                        c[1].update(c[0].position)), (st, nb))
        return st, nb, pot.energy(st.position, nb) + quantity.kinetic_energy(momentum=st.momentum, mass=st.mass)

    E = []
    for _ in range(4):
        st, nb, e = run(st, nb)
        assert not nb.did_buffer_overflow
        E.append(float(e))
        nb = jaxmd.allocate(nf, st.position)        # a new list into the same compiled function
    assert np.ptp(E) / len(at) < 1e-4


def test_cutoff_reads_the_model(cell):
    at, *_ = cell
    assert jaxmd.cutoff(str(MODEL)) == jaxmd.potential(str(MODEL), None, at.numbers).cutoff


def test_sparse_overflow_is_nan_not_truncation(cell):
    """An overflowed Sparse list is NaN, not a silently smaller energy."""
    at, _, box, disp, _, R = cell
    rc = jaxmd.cutoff(str(MODEL))
    pot = jaxmd.potential(str(MODEL), disp, at.numbers)
    nf = partition.neighbor_list(disp, box, rc, dr_threshold=1.0, format=partition.Sparse,
                                 fractional_coordinates=True, capacity_multiplier=0.5)
    nb = jaxmd.allocate(nf, R)
    assert bool(nb.did_buffer_overflow) and bool(pot.overflow(R, nb))
    assert np.isnan(float(pot.energy(R, nb))) and np.isnan(np.asarray(pot.force(R, nb))).all()


def test_sparse_lists_refuse_K(cell):
    at, _, box, disp, _, R = cell
    rc = jaxmd.cutoff(str(MODEL))
    _, nb = _list(disp, box, rc, R, partition.Sparse)
    with pytest.raises(ValueError, match="Sparse"):
        jaxmd.potential(str(MODEL), disp, at.numbers, K=48).energy(R, nb)
    with pytest.raises(ValueError, match="Dense"):
        jaxmd.capacity(R, nb, disp, rc)


def test_species_override_and_unknown_elements(cell):
    """species= per call: the same numbers give the same energy; a concrete unknown element
    raises, a traced one (inside jit) makes energy and forces NaN instead of a wrong species."""
    at, (E0, _, _), box, disp, _, R = cell
    pot = jaxmd.potential(str(MODEL), disp, at.numbers)
    _, nb = _list(disp, box, pot.cutoff, R)
    assert abs(float(pot.energy(R, nb, species=at.numbers)) - E0) < 1e-10 * len(at)
    carbon = np.array(at.numbers).copy()
    carbon[0] = 6
    with pytest.raises(ValueError, match="not in the model"):
        pot.energy(R, nb, species=carbon)
    f = jax.jit(lambda R, z: pot.energy(R, nb, species=z))
    g = jax.jit(lambda R, z: pot.force(R, nb, species=z))
    assert np.isnan(float(f(R, jnp.asarray(carbon)))) and np.isnan(np.asarray(g(R, jnp.asarray(carbon)))).all()
    assert abs(float(f(R, jnp.asarray(at.numbers))) - E0) < 1e-10 * len(at)


def test_overflow_flags_an_unknown_traced_species_and_cutoff_must_match(cell):
    at, _, box, disp, _, R = cell
    pot = jaxmd.potential(str(MODEL), disp, at.numbers)
    _, nb = _list(disp, box, pot.cutoff, R)
    carbon = np.array(at.numbers).copy()
    carbon[0] = 6
    f = jax.jit(lambda R, z: pot.overflow(R, nb, species=z))
    assert bool(f(R, jnp.asarray(carbon))) and not bool(f(R, jnp.asarray(at.numbers)))
    with pytest.raises(ValueError, match="cutoff"):
        jaxmd.potential(str(MODEL), disp, at.numbers, cutoff=4.0)
    model, meta, _ = aj_load(str(MODEL))
    assert jaxmd.cutoff(model, meta) == pot.cutoff
    with pytest.raises(ValueError, match="meta"):
        jaxmd.cutoff(model)


def _x64_off():
    """A context in which JAX defaults to float32 (the module enables x64)."""
    if hasattr(jax, "enable_x64"):
        return jax.enable_x64(False)
    from jax.experimental import disable_x64
    return disable_x64()


def test_float32_with_a_jax_md_list(cell):
    """A float32 model and positions with JAX-MD's own list: forces to float32 precision."""
    at, (_, F0, _), *_ = cell
    with _x64_off():
        box = jnp.asarray(at.cell.array.T, jnp.float32)
        disp, _ = space.periodic_general(box, fractional_coordinates=True)
        R = jnp.asarray(at.get_scaled_positions(), jnp.float32)
        pot = jaxmd.potential(str(MODEL), disp, at.numbers, dtype=jnp.float32)
        _, nb = _list(disp, box, pot.cutoff, R)
        pot = jaxmd.potential(str(MODEL), disp, at.numbers, dtype=jnp.float32, K=jaxmd.capacity(R, nb, disp, pot.cutoff))
        F = np.asarray(jax.jit(pot.force)(R, nb))
    assert F.dtype == np.float32
    np.testing.assert_allclose(F, F0, atol=2e-4 * np.abs(F0).max())


def test_min_image_ok():
    a = bulk("Si", "diamond", 5.43, cubic=True)
    assert not jaxmd.min_image_ok((a * (2, 2, 2)).cell.array, 6.0)     # 10.9 A < 12 A
    assert jaxmd.min_image_ok((a * (3, 3, 3)).cell.array, 6.0)
    assert not jaxmd.min_image_ok(np.array([[20.0, 0, 0], [19.0, 3.0, 0], [0, 0, 20.0]]), 6.0)   # sheared


def test_overflow_flag_marks_a_tight_K_and_an_overflowed_list(cell):
    """`overflow` is the explicit check behind the NaN: false with K >= the in-cutoff count,
    true (and energy NaN) when a row has more in-cutoff neighbours than K."""
    at, _, box, disp, _, R = cell
    rc = jaxmd.potential(str(MODEL), disp, at.numbers).cutoff
    _, nb = _list(disp, box, rc, R)
    k = jaxmd.capacity(R, nb, disp, rc)
    ok = jaxmd.potential(str(MODEL), disp, at.numbers, K=k)
    tight = jaxmd.potential(str(MODEL), disp, at.numbers, K=k - 8)
    assert not bool(jax.jit(ok.overflow)(R, nb)) and np.isfinite(float(ok.energy(R, nb)))
    assert bool(jax.jit(tight.overflow)(R, nb)) and np.isnan(float(tight.energy(R, nb)))
