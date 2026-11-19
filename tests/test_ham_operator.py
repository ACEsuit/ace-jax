"""H(k), S(k) and eigenvalues on the device against a host assembly of the predicted blocks."""
import jax

jax.config.update("jax_enable_x64", True)

import pathlib

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from ace_jax.basis import cg
from ace_jax.operators import bond
from ace_jax.operators.features import build_site_basis
from ace_jax.operators.offsite import OffsiteModel, canonical_pairs
from ace_jax.operators.onsite import OnsiteModel, parts_of, shell_pairs
from ace_jax.ham.operator import HamiltonianPredictor
from ace_jax.operators.orbitals import OrbitalBasis

CACHE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "ham" / "coupling_cache"
ORB = OrbitalBasis.of({14: (0, 1, 1, 2), 6: (0, 1)})
CH = ("H", "S")


@pytest.fixture(scope="module")
def models():
    site = build_site_basis([14, 6], order=2, totaldegree=5, lmax_feat=4, rcut=4.0, coupling_cache_dir=str(CACHE))
    basis = bond.BondBasis(site=site, rcut_bond=5.0, n_bond=4, n_cpl=2, l_cpl=2)
    rng = np.random.default_rng(0)
    on = {(z, s1, s2, L): 0.05 * rng.normal(size=(2, site.n(L, p) + (L == 0)))
          for z in ORB.species for s1, s2, l1, l2 in shell_pairs(ORB, z) for L, p in parts_of(l1, l2, s1 == s2)
          if site.n(L, p) + (L == 0)}
    for (z, s1, s2, L), c in on.items():          # make S well conditioned: identity plus small terms
        if L == 0 and s1 == s2:
            c[1, -1] = np.sqrt(2 * ORB.ls(z)[s1] + 1) * (1 if cg.real_cg(ORB.ls(z)[s1], ORB.ls(z)[s1], 0)[0, 0, 0] > 0 else -1)
    off = {}
    for zi in ORB.species:
        for zj in ORB.species:
            for s1, s2, l1, l2, same in canonical_pairs(ORB, zi, zj):
                p = 1 if (l1 + l2) % 2 == 0 else -1
                for L in cg.Ls(l1, l2):
                    k0, k1 = bond.n_features(basis, L, p)
                    if k0 + k1:
                        off[(zi, zj, s1, s2, L)] = 0.01 * rng.normal(size=(2, k0 + (k1 if same else 2 * k1)))
    return (OnsiteModel(basis=site, coef=on, orbitals=ORB, channels=CH),
            OffsiteModel(bond=basis, coef=off, orbitals=ORB, channels=CH))


def structure(seed=0):
    rng = np.random.default_rng(seed)
    cell = 4.6 * np.eye(3) + 0.3 * rng.normal(size=(3, 3))
    return rng.uniform(size=(4, 3)) @ cell, cell, np.array([14, 6, 14, 6])


def host_matrix(onsite, offsite, pos, cell, numbers, k, channel):
    """The same Bloch sum from the reference predictions, in numpy."""
    off = np.concatenate([[0], np.cumsum([ORB.n_orb(z) for z in numbers])])
    M = np.zeros((off[-1], off[-1]), complex)
    blocks = {**onsite.predict(pos, cell, numbers)[channel], **offsite.predict(pos, cell, numbers)[channel]}
    for (Rx, Ry, Rz, i, j), b in blocks.items():
        M[off[i]:off[i + 1], off[j]:off[j + 1]] += b * np.exp(2j * np.pi * np.dot(k, (Rx, Ry, Rz)))
    return M


def test_bloch_matrices_match_the_host_assembly(models):
    onsite, offsite = models
    pos, cell, numbers = structure()
    hp = HamiltonianPredictor(onsite, offsite)
    prep = hp.prepare(pos, cell, numbers)
    assert prep.n_orb == 2 * ORB.n_orb(14) + 2 * ORB.n_orb(6)
    for k in ((0.0, 0.0, 0.0), (0.31, -0.2, 0.45)):
        for c in CH:
            got, ref = np.asarray(hp.hk(prep, k, c)), host_matrix(onsite, offsite, pos, cell, numbers, k, c)
            assert np.abs(got - ref).max() < 1e-11 * max(1.0, np.abs(ref).max()), (k, c)
            assert np.abs(got - got.conj().T).max() < 1e-11                     # Hermitian: bond reversal is exact
    assert np.abs(np.asarray(hp.hk(prep, (0, 0, 0), "H")).imag).max() < 1e-12     # real at Gamma


def test_bands_match_scipy(models):
    onsite, offsite = models
    pos, cell, numbers = structure()
    hp = HamiltonianPredictor(onsite, offsite)
    prep = hp.prepare(pos, cell, numbers)
    kpts = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.13, 0.27, -0.4)]
    e = np.asarray(hp.bands(prep, kpts))
    assert e.shape == (3, prep.n_orb)
    for k, ek in zip(kpts, e):
        H = host_matrix(onsite, offsite, pos, cell, numbers, k, "H")
        S = host_matrix(onsite, offsite, pos, cell, numbers, k, "S")
        assert np.linalg.eigvalsh(S).min() > 0.2
        assert np.abs(ek - scipy.linalg.eigh(H, S, eigvals_only=True)).max() < 1e-9
    e0 = np.asarray(hp.bands(prep, [kpts[1]], overlap=None))[0]
    assert np.abs(e0 - np.linalg.eigvalsh(host_matrix(onsite, offsite, pos, cell, numbers, kpts[1], "H"))).max() < 1e-9
    # k and k + G give the same spectrum
    assert np.abs(np.asarray(hp.bands(prep, [(1.13, 0.27, -1.4)]))[0] - e[2]).max() < 1e-9


def test_models_must_share_the_site_basis(models):
    onsite, offsite = models
    other = build_site_basis([14, 6], order=2, totaldegree=5, lmax_feat=4, rcut=4.0, coupling_cache_dir=str(CACHE))
    with pytest.raises(ValueError, match="share one site basis"):
        HamiltonianPredictor(OnsiteModel(basis=other, coef=onsite.coef, orbitals=ORB, channels=CH), offsite)
    with pytest.raises(ValueError, match="channels"):
        HamiltonianPredictor(OnsiteModel(basis=onsite.basis, coef={}, orbitals=ORB, channels=("H",)), offsite)


def need_gpu_list():
    if jax.default_backend() != "gpu":
        pytest.skip("no CUDA backend")
    try:
        from ace_jax.eval.nlist import dense_graph
        dense_graph(np.zeros((1, 3)), 10 * np.eye(3), True, 2.0, 4, device="cuda")
    except Exception as e:                                  # the PyPI wheels are CPU-only
        pytest.skip(f"matscipy-neighbours has no CUDA backend ({type(e).__name__})")


def test_device_lists_give_the_same_matrices(models):
    """`prepare_device`: neighbour lists on the GPU, reverse slots, images and index lists
    derived there.  Same H(k) and S(k) as the host-list path, and every bond finds its reverse."""
    need_gpu_list()
    onsite, offsite = models
    hp = HamiltonianPredictor(onsite, offsite)
    for seed in (0, 2):
        pos, cell, numbers = structure(seed)
        a, b = hp.prepare(pos, cell, numbers), hp.prepare_device(pos, cell, numbers)
        assert bool(b.reverse_ok) and a.n_orb == b.n_orb
        for k in ((0.0, 0.0, 0.0), (0.31, -0.2, 0.45)):
            for c in CH:
                A, Bm = np.asarray(hp.hk(a, k, c)), np.asarray(hp.hk(b, k, c))
                assert np.abs(A - Bm).max() < 1e-10 * max(1.0, np.abs(A).max()), (seed, k, c)


def test_device_path_uploads_nothing_but_its_inputs(models):
    """After the first (compiling) call, positions to eigenvalues runs with JAX's host-to-device
    transfers disallowed: `prepare_device` uploads the positions, cell and species explicitly
    (and the neighbour-list library takes the positions), and nothing else goes up."""
    need_gpu_list()
    onsite, offsite = models
    hp = HamiltonianPredictor(onsite, offsite)
    pos, cell, numbers = structure()
    k = jax.device_put(jnp.asarray([0.13, 0.27, -0.4]))
    prep = hp.prepare_device(pos, cell, numbers)
    e0 = np.asarray(T_bands(hp, prep, k))
    with jax.transfer_guard_host_to_device("disallow"):
        moved = pos + 0.05 * np.random.default_rng(5).normal(size=pos.shape)
        prep = hp.prepare_device(moved, cell, numbers)               # new positions (not a rigid shift)
        e = T_bands(hp, prep, k)
    assert np.asarray(e).shape == e0.shape and np.abs(np.asarray(e) - e0).max() > 1e-6
    with pytest.raises(Exception, match="[Dd]isallowed"):            # the host-list path does upload
        with jax.transfer_guard_host_to_device("disallow"):
            hp.prepare(pos, cell, numbers)


def T_bands(hp, prep, k):
    """Eigenvalues at a k already on the device (`bands` takes host k-points)."""
    from ace_jax.ham.operator import _bloch, _eigvals
    H = _bloch(prep.on, prep.off, prep.lists, k, hp.channels.index("H"), prep.n_orb)
    S = _bloch(prep.on, prep.off, prep.lists, k, hp.channels.index("S"), prep.n_orb)
    return _eigvals(H, S)


def test_block_sums_and_batched_eigenvalues_match_the_host(models):
    """`BlockSum` on a dictionary of blocks and `eigvals` over a batch of k-points against a NumPy
    Bloch sum and SciPy, for a grid of k-points in batches that do not divide it."""
    import scipy.linalg

    from ace_jax.ham.operator import BlockSum, eigvals
    on, off = models
    pos, cell, numbers = structure()
    blocks = {c: {**off.predict(pos, cell, numbers)[c], **on.predict(pos, cell, numbers)[c]} for c in ("H", "S")}
    orb = on.orbitals
    o = np.concatenate([[0], np.cumsum([orb.n_orb(int(z)) for z in numbers])])
    ks = np.array([[a, b, 0.3] for a in (0.0, 0.25, 0.5) for b in (0.0, 0.1, 0.7)])
    bs = BlockSum(list(blocks["S"]), numbers, orb)
    H, S = bs.matrices(blocks["H"], ks, chunk=4), bs.matrices(blocks["S"], ks, chunk=4)
    S = S + 3.0 * np.eye(bs.n_orb)                       # the random model's overlap is not positive definite
    for q, k in enumerate(ks):
        ref = np.zeros((o[-1], o[-1]), complex)
        for (Rx, Ry, Rz, i, j), b in blocks["H"].items():
            ref[o[i]:o[i + 1], o[j]:o[j + 1]] += np.exp(2j * np.pi * np.dot(k, (Rx, Ry, Rz))) * b
        ref = 0.5 * (ref + ref.conj().T)
        assert np.abs(np.asarray(H[q]) - ref).max() < 1e-10 * max(1.0, np.abs(ref).max())
    e = np.asarray(eigvals(H, S, chunk=4))
    want = np.array([scipy.linalg.eigh(np.asarray(h), np.asarray(s), eigvals_only=True) for h, s in zip(H, S)])
    assert e.shape == want.shape and np.abs(e - want).max() < 1e-8 * max(1.0, np.abs(want).max())
    from ace_jax.ham.operator import block_bands
    shifted = {k: v + (3.0 * np.eye(len(v)) if k[:3] == (0, 0, 0) and k[3] == k[4] else 0.0) for k, v in blocks["S"].items()}
    assert np.abs(block_bands(bs, blocks["H"], shifted, ks, chunk=4) - want).max() < 1e-8 * max(1.0, np.abs(want).max())


@pytest.mark.parametrize("overlap", [None, "S"])
def test_spectrum_is_differentiable_in_the_coefficients(models, overlap):
    """`spectrum_fn`: the eigenvalues as one program of the coefficients.  Its value is `bands`,
    its gradient of a spectral loss agrees with a central difference along a random direction in
    every coefficient array (on-site and off-site, H and S), and a few steps of gradient descent
    on the distance to a target spectrum reduce it (a crude fixed step: only the direction is tested)."""
    import jax
    on, off = models
    env = off.bond._replace(r_env=4.0, n_env=2, l_env=2, n_ebond=2, l_ebond=2)       # with the bond-environment group
    rng = np.random.default_rng(3)
    coef = dict(off.coef)
    from ace_jax.operators.offsite import OffsiteModel, canonical_pairs
    from ace_jax.basis import cg as CG
    for za in on.orbitals.species:
        for zb in on.orbitals.species:
            for s1, s2, l1, l2, same in canonical_pairs(on.orbitals, za, zb):
                p = 1 if (l1 + l2) % 2 == 0 else -1
                for L in CG.Ls(l1, l2):
                    k0, k1 = bond.n_features(env, L, p)
                    k = k0 + (k1 if same else 2 * k1) + bond.n_env_features(env, L, p, same=same)
                    if k:
                        coef[(za, zb, s1, s2, L)] = 0.05 * rng.normal(size=(2, k))
    hp = HamiltonianPredictor(on, OffsiteModel(bond=env, coef=coef, orbitals=on.orbitals, channels=on.channels))
    pos, cell, numbers = structure()
    prep = hp.prepare(pos, cell, numbers)
    ks = np.array([[0.1, 0.2, -0.3], [0.37, 0.0, 0.21]])
    shift = 6.0 if overlap else 0.0
    fn, params = hp.spectrum_fn(prep, ks, overlap=overlap, overlap_shift=shift)
    e = np.asarray(fn(params))
    if overlap is None:
        assert np.allclose(e, np.asarray(hp.bands(prep, ks, overlap=None)), atol=1e-9)
    assert np.isfinite(e).all() and e.shape == (2, prep.n_orb)
    target = e + 0.1 * np.random.default_rng(1).normal(size=e.shape)
    loss = lambda p: jax.numpy.sum((fn(p) - target) ** 2)
    val, grad = jax.value_and_grad(loss)(params)
    leaves, _ = jax.tree.flatten(grad)
    assert len(leaves) > 10 and all(np.isfinite(np.asarray(x)).all() for x in leaves)
    assert sum(float(np.abs(np.asarray(x)).max() > 0) for x in leaves) > 0.5 * len(leaves)    # most arrays matter
    r = np.random.default_rng(2)
    direction = jax.tree.map(lambda x: r.normal(size=x.shape), params)
    step = lambda t: jax.tree.map(lambda x, d: x + t * d, params, direction)
    h = 1e-5
    fd = (float(loss(step(h))) - float(loss(step(-h)))) / (2 * h)
    an = sum(float(np.vdot(np.asarray(g), np.asarray(d))) for g, d in zip(leaves, jax.tree.leaves(direction)))
    assert abs(fd - an) < 1e-5 * max(1.0, abs(an)), (fd, an)
    p, lr = params, 0.2 * float(val) / sum(float((np.asarray(x) ** 2).sum()) for x in leaves)
    for _ in range(5):
        p = jax.tree.map(lambda x, g: x - lr * g, p, jax.grad(loss)(p))
    assert float(loss(p)) < 0.97 * float(val)


def _env_predictor(models):
    """A random model with the site-coupled and bond-environment groups, H and S."""
    from ace_jax.basis import cg as CG
    from ace_jax.operators.offsite import OffsiteModel, canonical_pairs
    on, off = models
    env = off.bond._replace(r_env=4.0, n_env=2, l_env=2, n_ebond=2, l_ebond=2)
    rng = np.random.default_rng(3)
    coef = {}
    for za in on.orbitals.species:
        for zb in on.orbitals.species:
            for s1, s2, l1, l2, same in canonical_pairs(on.orbitals, za, zb):
                p = 1 if (l1 + l2) % 2 == 0 else -1
                for L in CG.Ls(l1, l2):
                    k0, k1 = bond.n_features(env, L, p)
                    k = k0 + (k1 if same else 2 * k1) + bond.n_env_features(env, L, p, same=same)
                    if k:
                        coef[(za, zb, s1, s2, L)] = 0.05 * rng.normal(size=(2, k))
    return HamiltonianPredictor(on, OffsiteModel(bond=env, coef=coef, orbitals=on.orbitals, channels=on.channels))


@pytest.mark.parametrize("overlap", [None, "S"])
def test_spectrum_is_differentiable_in_the_positions(models, overlap):
    """`bands_fn`: the eigenvalues as one program of the atomic positions (what forces need).  Its
    value is `bands`; the gradient of a band-energy-like sum agrees with central differences of
    the program itself and of a fresh evaluation with new neighbour lists; and it sums to zero
    over the atoms (a rigid translation changes nothing)."""
    import jax
    hp = _env_predictor(models)
    pos, cell, numbers = structure()
    ks = np.array([[0.1, 0.2, -0.3], [0.37, 0.0, 0.21]])
    shift = 6.0 if overlap else 0.0
    fn, p0 = hp.bands_fn(pos, cell, numbers, ks, overlap=overlap, overlap_shift=shift)
    e = np.asarray(fn(p0))
    assert np.isfinite(e).all()
    if overlap is None:
        assert np.allclose(e, np.asarray(hp.bands(hp.prepare(pos, cell, numbers), ks, overlap=None)), atol=1e-9)
    w = np.random.default_rng(5).uniform(size=e.shape)                     # occupations of a band energy
    energy = lambda p: jax.numpy.sum(w * fn(p))
    grad = np.asarray(jax.grad(energy)(p0))
    assert grad.shape == pos.shape and np.isfinite(grad).all() and np.abs(grad).max() > 1e-3
    assert np.abs(grad.sum(axis=0)).max() < 1e-8 * np.abs(grad).max()       # translation invariance
    d = np.random.default_rng(6).normal(size=pos.shape)
    h = 1e-5
    fd = (float(energy(p0 + h * d)) - float(energy(p0 - h * d))) / (2 * h)
    assert abs(fd - float((grad * d).sum())) < 1e-6 * max(1.0, abs(fd)), (fd, float((grad * d).sum()))

    def fresh(x):                                                         # new lists at the moved positions
        f, q = hp.bands_fn(x, cell, numbers, ks, overlap=overlap, overlap_shift=shift)
        return float((w * np.asarray(f(q))).sum())
    h = 1e-4
    fd2 = (fresh(pos + h * d) - fresh(pos - h * d)) / (2 * h)
    assert abs(fd2 - float((grad * d).sum())) < 1e-5 * max(1.0, abs(fd2)), (fd2, float((grad * d).sum()))


def test_band_energy_gradient_by_the_trace_form(models):
    """`band_energy_fn`: E_band = sum f e and its gradient from tr[rho dH - W dS], without
    differentiating the eigensolver.  It equals the gradient through the eigensolver
    (`bands_fn`) and a central difference; with occupations as a function of the eigenvalues it
    is the derivative at those occupations held fixed; and on a crystal with degenerate states
    at Gamma it is finite, where it is tested against a central difference with every state
    occupied (a smooth function of the positions whatever the degeneracies)."""
    import jax
    hp = _env_predictor(models)
    pos, cell, numbers = structure()
    ks = np.array([[0.1, 0.2, -0.3], [0.37, 0.0, 0.21]])
    occ = np.zeros((2, hp.prepare(pos, cell, numbers).n_orb)); occ[:, :20] = 2.0; occ[:, 20] = 0.7
    fn, p0 = hp.band_energy_fn(pos, cell, numbers, ks, occ, overlap_shift=6.0)
    out = fn(p0)
    E, g, e = (np.asarray(x) for x in (out.energy, out.gradient, out.eigenvalues))
    bands, _ = hp.bands_fn(pos, cell, numbers, ks, overlap_shift=6.0)
    through = lambda p: jax.numpy.sum(0.5 * occ * bands(p))
    assert np.allclose(e, np.asarray(bands(p0)), atol=1e-9) and abs(E - float(through(p0))) < 1e-9
    assert np.abs(g - np.asarray(jax.grad(through)(p0))).max() < 1e-8 * np.abs(g).max()
    assert np.abs(g.sum(axis=0)).max() < 1e-8 * np.abs(g).max()
    d = np.random.default_rng(6).normal(size=pos.shape)
    h = 1e-5
    fd = (float(fn(p0 + h * d).energy) - float(fn(p0 - h * d).energy)) / (2 * h)
    assert abs(fd - float((g * d).sum())) < 1e-6 * max(1.0, abs(fd))
    # occupations from the eigenvalues: held fixed in the derivative
    mu = 0.5 * (e[:, 20] + e[:, 21]).mean()
    fermi = lambda x: 2.0 / (1.0 + jax.numpy.exp((x - mu) / 0.05))
    f2, _ = hp.band_energy_fn(pos, cell, numbers, ks, fermi, overlap_shift=6.0)
    fixed, _ = hp.band_energy_fn(pos, cell, numbers, ks, np.asarray(fermi(e)), overlap_shift=6.0)
    assert np.allclose(np.asarray(f2(p0).gradient), np.asarray(fixed(p0).gradient), atol=1e-10)
    # a crystal: zincblende, degenerate states at Gamma
    a = 4.4
    zb_cell = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    zb_pos = np.array([[0.0, 0, 0], [0.25 * a] * 3])
    zb_numbers = np.array([14, 6])
    gam = np.zeros((1, 3))
    full, q0 = hp.band_energy_fn(zb_pos, zb_cell, zb_numbers, gam, 1.0, overlap_shift=6.0)
    out0 = full(q0)
    E0, g0, e0 = (np.asarray(x) for x in (out0.energy, out0.gradient, out0.eigenvalues))
    assert (np.diff(e0[0]) < 1e-8).sum() >= 2                              # degenerate levels are there
    assert np.isfinite(g0).all() and np.abs(g0.sum(axis=0)).max() < 1e-8 * max(1.0, np.abs(g0).max())
    d = np.random.default_rng(7).normal(size=zb_pos.shape)
    fd = (float(full(q0 + h * d).energy) - float(full(q0 - h * d).energy)) / (2 * h)
    assert abs(fd - float((g0 * d).sum())) < 1e-6 * max(1.0, abs(fd), np.abs(g0).max())


def test_stress_is_the_strain_derivative_of_the_band_energy(models):
    """The stress of `band_energy_fn` against central differences of E_band with positions and
    cell strained together (fresh neighbour lists), every component; it is symmetric."""
    hp = _env_predictor(models)
    pos, cell, numbers = structure()
    ks = np.array([[0.1, 0.2, -0.3], [0.37, 0.0, 0.21]])
    occ = np.zeros(hp.prepare(pos, cell, numbers).n_orb); occ[:20] = 2.0
    fn, p0 = hp.band_energy_fn(pos, cell, numbers, ks, occ, overlap_shift=6.0)
    out = fn(p0)
    stress, V = np.asarray(out.stress), abs(np.linalg.det(cell))
    assert stress.shape == (3, 3) and np.allclose(stress, stress.T) and np.abs(stress).max() > 1e-4

    def energy(eps):
        A = np.eye(3) + eps
        f, q = hp.band_energy_fn(pos @ A.T, cell @ A.T, numbers, ks, occ, overlap_shift=6.0)
        return float(f(q).energy)
    h = 1e-5
    for a in range(3):
        for b in range(a, 3):
            eps = np.zeros((3, 3)); eps[a, b] += 0.5; eps[b, a] += 0.5           # a symmetric strain
            fd = (energy(h * eps) - energy(-h * eps)) / (2 * h) / V
            assert abs(fd - stress[a, b]) < 1e-6 * max(1.0, np.abs(stress).max()), (a, b, fd, stress[a, b])


def test_band_energy_from_device_lists(models):
    """`band_energy_device` (lists on the GPU): the energy, forces, stress and eigenvalues of the
    host-list `band_energy_fn`, and again after the atoms move (new lists, the same compiled
    program).  Orthonormal basis: the random model's overlap is not positive definite."""
    import jax
    from ace_jax.eval.nlist import have_matscipy_neighbours
    if jax.default_backend() != "gpu" or not have_matscipy_neighbours():
        pytest.skip("needs a GPU and matscipy-neighbours")
    hp = _env_predictor(models)
    pos, cell, numbers = structure()
    ks = np.array([[0.1, 0.2, -0.3], [0.37, 0.0, 0.21]])
    occ = np.zeros(hp.prepare(pos, cell, numbers).n_orb); occ[:20] = 2.0
    try:
        dev = hp.band_energy_device(pos, cell, numbers, ks, occ, overlap=None)
    except RuntimeError as e:
        pytest.skip(f"matscipy-neighbours has no CUDA backend ({type(e).__name__})")
    moved = pos + 0.05 * np.random.default_rng(8).normal(size=pos.shape)
    for x in (pos, moved):
        dev = hp.band_energy_device(x, cell, numbers, ks, occ, overlap=None)
        fn, q = hp.band_energy_fn(x, cell, numbers, ks, occ, overlap=None)
        ref = fn(q)
        assert abs(float(dev.energy) - float(ref.energy)) < 1e-8 * abs(float(ref.energy))
        assert np.abs(np.asarray(dev.eigenvalues) - np.asarray(ref.eigenvalues)).max() < 1e-8
        assert np.abs(np.asarray(dev.gradient) - np.asarray(ref.gradient)).max() < 1e-7 * np.abs(np.asarray(ref.gradient)).max()
        assert np.abs(np.asarray(dev.stress) - np.asarray(ref.stress)).max() < 1e-7 * np.abs(np.asarray(ref.stress)).max()


def test_contractions_are_traced_at_full_precision(models):
    """Every dot product of the predictors is traced at the highest matmul precision: the GPU's
    float32 default (TF32) made the on-site blocks wrong by 0.5 eV and the band energy of a
    64-atom cell by a thousand eV.  Read off the traced programs (no GPU needed)."""
    import jax

    from ace_jax.ham import operator as O
    from ace_jax.operators import offsite_fast as OF
    hp = _env_predictor(models)
    pos, cell, numbers = structure()
    prep = hp.prepare(pos, cell, numbers)
    F, atoms, lay, env = prep.inputs
    b = hp.offsite.model.bond
    G, n_orb = hp._host_lists(pos, cell, numbers)
    hs_static = (hp.offsite.static(), n_orb, 0, 1, b.rcut_bond)
    programs = {
        "H(k), S(k) of positions and strain": jax.make_jaxpr(lambda p, e: O._host_hs(b.site, hp.onsite.readout, hp.offsite.packs, p, e, G, np.zeros((1, 3)), hs_static))(
            np.asarray(pos), np.zeros((3, 3))),
        "site features": jax.make_jaxpr(lambda r: b.site.features(r, np.zeros(5, np.int32), np.ones(5, np.int32), np.zeros(5, np.int32), 1))(np.ones((5, 3))),
        "on-site readout": jax.make_jaxpr(lambda f: hp.onsite.readout.blocks(f, 0, atoms[0]))(F),
        "off-site blocks": jax.make_jaxpr(lambda f: OF._blocks_impl(hp.offsite.packs, f, lay, b.rcut_bond, hp.offsite.static(), env))(F),
        "Bloch sum": jax.make_jaxpr(lambda k: O._bloch_impl(prep.on, prep.off, prep.lists, k, 0, prep.n_orb))(np.zeros(3)),
        "eigensolver": jax.make_jaxpr(O._geig)(np.eye(4, dtype=complex), np.eye(4, dtype=complex)),
    }
    seen = 0
    for name, pr in programs.items():
        text = str(pr)
        n = text.count("dot_general")
        seen += n
        # a dot product at the default precision prints no precision at all
        assert n == text.count("precision=(Precision.HIGHEST, Precision.HIGHEST)"), f"{name}: a dot product at the default precision"
    assert seen > 10
