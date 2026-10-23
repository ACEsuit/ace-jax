"""JAX-MD's integrators and optimizers under a plain jax.vmap over a padded batch (`ace_jax.batch`):
`BatchCalculator.energy_fn` with `BatchNeighbours` lists.  The energy and forces of every structure equal
ACECalculator's, before and after the atoms move and the lists are rebuilt; vmapped FIRE, NVE and Langevin
equal the same runs one structure at a time; padding atoms stay inert (no momentum, no centre-of-mass
drift of the live atoms)."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase.build import bulk

from ace_jax import batch
from ace_jax.batch import BatchCalculator
from ace_jax.calc.point import ACECalculator
from conftest import FIXTURE_DIR, require_optional

MODEL = str(FIXTURE_DIR / "si_fitted.npz")
MASS = 28.0855
DT = 1.0 / 10.18                                   # 1 fs in ASE time units


@pytest.fixture(scope="module")
def jax_md():
    return require_optional("jax_md")


def structures():
    """Mixed sizes padded into one batch: a 2-atom primitive cell (below the cutoff), a 6-atom vacancy
    cell, an 8-atom cell, a 16-atom sheared cell and a slab (pbc T T F)."""
    prim = bulk("Si", "diamond", a=5.43)
    cub = bulk("Si", "diamond", a=5.43, cubic=True)
    vac = cub.copy()
    del vac[[0, 3]]
    sheared = cub.repeat((2, 1, 1))
    sheared.set_cell(sheared.cell.array + [[0, 0, 0], [1.1, 0, 0], [0.4, 0.7, 0]], scale_atoms=True)
    slab = cub.repeat((1, 1, 2))
    slab.center(vacuum=5.0, axis=2)
    slab.pbc = (True, True, False)
    out = []
    for k, at in enumerate((prim, vac, cub, sheared, slab)):
        at = at.copy()
        at.rattle(0.05, seed=k)
        out.append(at)
    return out


def reference(ats, positions, **kw):
    """ACECalculator energies and forces of `ats` at `positions` (per-structure (n, 3))."""
    calc = ACECalculator(MODEL, **kw)
    E, F = [], []
    for at, R in zip(ats, positions):
        at = at.copy()
        at.positions = np.asarray(R)
        at.calc = calc
        E.append(at.get_potential_energy())
        F.append(at.get_forces())
    return np.array(E), F


def check_against_calculator(bc, nb, nl, ats, R, **kw):
    energy = bc.energy_fn()
    E, G = jax.jit(jax.vmap(jax.value_and_grad(lambda r, n: energy(r, neighbor=n))))(R, nl)
    Eref, Fref = reference(ats, nb.unpad(R), **kw)
    assert np.abs(np.asarray(E) - Eref).max() <= 1e-12 * np.abs(Eref).max()
    F = nb.unpad(-G)
    for f, fr in zip(F, Fref):
        assert np.abs(np.asarray(f) - fr).max() <= 1e-11 * max(np.abs(fr).max(), 1.0)
    assert np.all(np.asarray(G)[~np.asarray(nb.live)] == 0)                # padding: no force


def test_energy_fn_matches_ACECalculator_per_structure():
    ats = structures()
    bc = BatchCalculator(MODEL)
    nb = bc.neighbour_lists(ats, skin=1.0)
    assert nb.positions.shape == (5, 16, 3)
    nl = nb.allocate()
    check_against_calculator(bc, nb, nl, ats, nb.positions)


def run_chunks(nb, nl, step, state, chunks, every):
    """`chunks` jitted chunks of `every` vmapped steps, the lists checked (and rebuilt) between chunks."""
    for _ in range(chunks):
        state = step(state, nl)
        nl, _ = nb.update(state.position, nl)
    return state, nl


def test_vmapped_fire_relaxes_each_structure(jax_md):
    from jax_md import minimize, space
    ats = structures()
    bc = BatchCalculator(MODEL)
    nb = bc.neighbour_lists(ats, skin=0.5)
    nl = nb.allocate()
    _, shift = space.free()
    init, apply = minimize.fire_descent(bc.energy_fn(), shift, dt_start=0.05, dt_max=0.2)
    chunk = lambda st, n: jax.lax.fori_loop(0, 50, lambda i, s: apply(s, neighbor=n), st)
    step = jax.jit(jax.vmap(chunk))
    state = jax.jit(jax.vmap(lambda R, n: init(R, neighbor=n)))(nb.positions, nl)
    state, nl = run_chunks(nb, nl, step, state, 8, 50)
    live = np.asarray(nb.live)
    assert np.array_equal(np.asarray(state.position)[~live], np.asarray(nb.positions)[~live])   # padding: still
    check_against_calculator(bc, nb, nl, ats, state.position)        # lists still right after the moves
    Eref, Fref = reference(ats, nb.unpad(state.position))
    assert max(np.abs(f).max() for f in Fref) < 1e-2                 # relaxed
    # one structure at a time, the same steps: the same result (FIRE's dt, alpha, power are per structure)
    nl1 = nb.allocate(nb.positions)
    for s in range(len(ats)):
        n = jax.tree.map(lambda a: a[s], nl1)
        st = init(nb.positions[s], neighbor=n)
        for _ in range(8):
            st = jax.jit(chunk)(st, n)
            moved = float(jnp.where(n.live, jnp.linalg.norm(st.position - n.position0, axis=-1), 0).max())
            if moved > nb.skin / 2:                               # rebuild this one as the batch would
                full = nb.allocate(jnp.asarray(state.position).at[s].set(st.position))
                n = jax.tree.map(lambda a: a[s], full)
        assert np.abs(np.asarray(st.position) - np.asarray(state.position[s])).max() < 1e-6


def test_vmapped_nve_padding_inert_and_rebuilds(jax_md):
    from jax_md import simulate, space
    ats = structures()
    bc = BatchCalculator(MODEL)
    _, shift = space.free()
    init, apply = simulate.nve(bc.energy_fn(), shift, dt=DT)
    finals = {}
    for skin in (0.3, 3.0):                       # 0.3: rebuilt several times; 3.0: never (the vacancy cell and the
        #                                           slab rearrange: atoms move ~1 A in the 200 fs)
        nb = bc.neighbour_lists(ats, skin=skin)
        nl = nb.allocate()
        live = nb.live
        p0 = batch.initial_momenta(jax.random.PRNGKey(0), live, 0.05, MASS)
        state = jax.jit(jax.vmap(lambda R, p, n: init(None, R, kT=0.0, mass=MASS, momenta=p, neighbor=n)))(
            nb.positions, p0, nl)
        energy = bc.energy_fn()
        total = jax.jit(jax.vmap(lambda st, n: energy(st.position, neighbor=n)
                                 + batch.kinetic_energy(st.momentum, MASS, n.live)))
        E0 = np.asarray(total(state, nl))
        step = jax.jit(jax.vmap(lambda st, n: jax.lax.fori_loop(0, 20, lambda i, s: apply(s, neighbor=n), st)))
        state, nl = run_chunks(nb, nl, step, state, 10, 20)
        assert (nb.rebuilds > 1) == (skin < 1.0), nb.rebuilds
        p = np.asarray(state.momentum)
        assert np.all(p[~np.asarray(live)] == 0)                                   # padding: no momentum
        assert np.abs(np.where(np.asarray(live)[..., None], p, 0).sum(1)).max() < 1e-10   # no live COM drift
        assert np.abs(np.asarray(total(state, nl)) - E0).max() < 5e-3              # 200 fs, 1 fs steps
        check_against_calculator(bc, nb, nl, ats, state.position)
        finals[skin] = np.asarray(state.position)
    # the skin changes only the summation order: roundoff, amplified in 200 fs of the rearranging cells to ~1e-5 A
    assert np.abs(finals[0.3] - finals[3.0]).max() < 1e-4


def test_vmapped_langevin_matches_single_and_temperature(jax_md):
    from jax_md import simulate, space
    ats = structures()
    bc = BatchCalculator(MODEL)
    nb = bc.neighbour_lists(ats, skin=1.0)
    nl = nb.allocate()
    _, shift = space.free()
    init, apply = simulate.nvt_langevin(bc.energy_fn(), shift, dt=DT, kT=0.025, gamma=0.1)
    keys = jax.random.split(jax.random.PRNGKey(1), len(ats))
    p0 = batch.initial_momenta(keys[0], nb.live, 0.025, MASS)

    def run(key, R, p, n):
        st = init(key, R, mass=MASS, momenta=p, neighbor=n)
        return jax.lax.fori_loop(0, 30, lambda i, s: apply(s, neighbor=n), st)
    vm = jax.jit(jax.vmap(run))(keys, nb.positions, p0, nl)
    for s in range(len(ats)):
        one = jax.jit(run)(keys[s], nb.positions[s], p0[s], jax.tree.map(lambda a: a[s], nl))
        assert np.abs(np.asarray(one.position) - np.asarray(vm.position[s])).max() < 1e-12
    clean = batch.zero_padding(vm, nb.live)
    assert np.all(np.asarray(clean.momentum)[~np.asarray(nb.live)] == 0)
    T = np.asarray(batch.temperature(clean.momentum, clean.mass, nb.live))
    assert T.shape == (len(ats),) and np.all(np.isfinite(T)) and np.all(T > 0)


def test_initial_momenta_and_temperature():
    live = jnp.asarray(np.arange(64)[None, :] < np.array([[64], [40], [3]]))
    p = batch.initial_momenta(jax.random.PRNGKey(2), live, 0.1, 2.0)
    assert np.all(np.asarray(p)[~np.asarray(live)] == 0)
    assert np.abs(np.asarray(p).sum(1)).max() < 1e-12                          # COM per structure
    T = np.asarray(batch.temperature(p, 2.0, live))
    assert abs(T[0] - 0.1) < 0.05 and np.all(T > 0)                           # 64 atoms: near kT
    m = jnp.where(live, 1.0 + jnp.arange(64) / 64.0, 1.0)                     # per-atom masses, (B, N)
    p = batch.initial_momenta(jax.random.PRNGKey(3), live, 0.1, m)
    assert np.abs(np.asarray(p).sum(1)).max() < 1e-12


def test_needs_rebuild_per_structure():
    ats = structures()
    bc = BatchCalculator(MODEL)
    nb = bc.neighbour_lists(ats, skin=0.4)
    nl = nb.allocate()
    R = nb.positions.at[2, 0, 0].add(0.25)                                    # one atom of structure 2
    assert nb.needs_rebuild(R, nl).tolist() == [False, False, True, False, False]
    assert nb.update(nb.positions, nl)[1] is False and nb.update(R, nl)[1] is True


def test_vmapped_md_with_the_kernel(jax_md):
    """ffi=True: the C++ kernel under jax.vmap (its rows flattened into one call).  Energies and forces equal
    ACECalculator(ffi=True)'s to roundoff (both read one radial table); vmapped NVE equals the per-structure
    runs and stays on the JAX model's trajectory to the table's accuracy."""
    import os
    import shutil
    if jax.default_backend() != "cpu" or shutil.which("c++") is None:
        if os.environ.get("ACEJAX_REQUIRE_OPTIONAL") and jax.default_backend() == "cpu":
            pytest.fail("no C++ compiler")
        pytest.skip("the C++ kernel needs a CPU and a compiler")
    from jax_md import simulate, space
    ats = structures()
    _, shift = space.free()
    finals = {}
    for ffi in (True, False):
        bc = BatchCalculator(MODEL, ffi=ffi)
        assert (bc.ffi is not None) == ffi
        nb = bc.neighbour_lists(ats, skin=1.0)
        nl = nb.allocate()
        if ffi:
            check_against_calculator(bc, nb, nl, ats, nb.positions, ffi=True)
        init, apply = simulate.nve(bc.energy_fn(), shift, dt=DT)
        p0 = batch.initial_momenta(jax.random.PRNGKey(0), nb.live, 0.05, MASS)
        run = lambda R, p, n: jax.lax.fori_loop(0, 50, lambda i, s: apply(s, neighbor=n),
                                                init(None, R, kT=0.0, mass=MASS, momenta=p, neighbor=n))
        vm = jax.jit(jax.vmap(run))(nb.positions, p0, nl)
        if ffi:
            for s in range(len(ats)):
                one = jax.jit(run)(nb.positions[s], p0[s], jax.tree.map(lambda a: a[s], nl))
                assert np.abs(np.asarray(one.position) - np.asarray(vm.position[s])).max() < 1e-12
        finals[ffi] = np.asarray(vm.position)
    assert np.abs(finals[True] - finals[False]).max() < 1e-5      # radial table: forces ~1e-8 of max|F|
