# Reference for the BUILT basis: ACEpotentials' design matrix for the bases
# ace-jax builds in Python (tests/test_built_basis_parity.py).
#
#   julia --project=julia julia/built_basis_reference.jl [outdir]
#
# export_model.jl exports a model ACEpotentials built AND coupled, so the parity
# tests on it check ace-jax's evaluator, never its builder (`build_basis`: spec,
# coupling library, column order).  That gap hid #57.  Here each case is the
# `ace_model` that `ace_jax.basis.build_model` reproduces -- solid harmonics,
# analytic radials, the ace1 level TotalDegree(NZ, 1/wL) -- and the file holds:
#   * the radial parameters (random Glorot init: any generic radial will do), so
#     the Python test can graft them onto its own build and leave only the
#     builder's structure (Rnl/Ylm spec, aspec, aa_specs, A2B) under test;
#   * the basis energies, forces and virials (the unweighted design-matrix rows)
#     of a few rattled bulk cells, every species present in each;
#   * the per-row nnll and the smoothness prior, which the test matches by body.
# CI: .github/workflows/julia-parity.yml regenerates these and runs the test.

using ACEpotentials, NPZ, JSON, StaticArrays, LinearAlgebra, Random, Printf
using AtomsBase, AtomsBuilder, Unitful, Lux
M = ACEpotentials.Models
include(joinpath(@__DIR__, "export_arrays.jl"))

const OUTDIR = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "fixtures")
const WL = 1.5
const RCUT = 5.5
# (elements, order, totaldegree).  Each interleaves body orders in build_spec's
# mb_spec (the #57 trigger).  o4d12 is the smallest Si basis with a block of
# multiplicity > 1 (one block, of 2); SiGe adds species blocks and per-pair
# radials.
const CASES = [((:Si,), 3, 10), ((:Si,), 4, 12), ((:Si, :Ge), 3, 6)]

function configs(elements; seed = 7)
    rng = MersenneTwister(seed)
    zs = [AtomsBase.atomic_number(ChemicalSpecies(el)) for el in elements]
    out = []
    for (k, sys) in enumerate((bulk(:Si, cubic = true),
                               bulk(:Si, cubic = true) * (2, 1, 1),
                               bulk(:Si) * (2, 2, 2)))
        n = length(sys)
        # every species in every cell (round robin), then a seeded shuffle so the
        # arrangement differs between cells
        Z = shuffle(rng, [zs[mod1(i, length(zs))] for i in 1:n])
        X = [ustrip.(u"Å", AtomsBase.position(sys, i)) .+ 0.2 .* randn(rng, 3) for i in 1:n]
        C = [ustrip.(u"Å", v) .* (1 .+ 0.03 .* randn(rng, 3)) for v in AtomsBase.cell_vectors(sys)]
        ats = [AtomsBase.Atom(Z[i], SVector{3}(X[i]) .* u"Å") for i in 1:n]
        push!(out, AtomsBase.FlexibleSystem(ats, Tuple(c .* u"Å" for c in C), (true, true, true)))
    end
    return out
end

for (elements, order, totaldegree) in CASES
    NZ = length(elements)
    ri = M._default_rin0cuts(elements)
    ri = (x -> (rin = x.rin, r0 = x.r0, rcut = RCUT)).(ri)
    raw = M.ace_model(; elements = elements, order = order, Ytype = :solid,
                      level = M.TotalDegree(1.0 * NZ, 1 / WL), max_level = totaldegree,
                      pair_maxn = totaldegree, rin0cuts = ri,
                      init_WB = :glorot_normal, init_Wpair = :glorot_normal)
    ps, st = Lux.setup(MersenneTwister(1234), raw)
    calc = M.ACEPotential(raw, ps, st)
    m = raw
    @assert !_is_spline(m.rbasis) && _is_spline(m.pairbasis)
    name = join(string.(elements)) * "_o$(order)d$(totaldegree)"

    Wnlq, pA, pB, pC = analytic_arrays(m.rbasis, ps.rbasis)
    pcoefs, px0, ph, pn = spline_arrays(m.pairbasis)
    penv, penv_kind = env1sr_params(m.pairbasis)
    @assert penv_kind == "poly1sr"
    D = Dict{String, Any}(
        "rnl_Wnlq" => Wnlq, "polys_A" => pA, "polys_B" => pB, "polys_C" => pC,
        "rnl_transform" => transform_params(m.rbasis), "rnl_envelope" => env2sx_params(m.rbasis),
        "pair_transform" => transform_params(m.pairbasis), "pair_envelope" => penv,
        "pair_spline_coefs" => pcoefs,
        "rnl_spec" => Int32[b[k] for b in m.rbasis.spec, k in (:n, :l)],   # (n_rnl, 2)
        "gamma" => collect(diag(M.algebraic_smoothness_prior(m))),
    )
    nnll = [[[b.n, b.l] for b in bb] for bb in M.get_nnll_spec(m.tensor)]
    n_B, len_basis = length(nnll), M.length_basis(m)

    for (k, sys) in enumerate(configs(elements))
        efv = M.energy_forces_virial_basis(sys, calc)
        n = length(sys)
        D["cfg$(k)_Z"] = Int32[AtomsBase.atomic_number(sys, i) for i in 1:n]
        D["cfg$(k)_pos"] = reduce(hcat, [ustrip.(u"Å", AtomsBase.position(sys, i)) for i in 1:n])  # (3, n)
        D["cfg$(k)_cell"] = reduce(hcat, [ustrip.(u"Å", v) for v in AtomsBase.cell_vectors(sys)]) # columns
        D["cfg$(k)_E"] = ustrip.(efv.energy)                                                     # (L,)
        D["cfg$(k)_F"] = [ustrip(efv.forces[i, l][a]) for i in 1:n, a in 1:3, l in 1:len_basis]  # (n, 3, L)
        D["cfg$(k)_V"] = [ustrip(efv.virial[l][a, b]) for a in 1:3, b in 1:3, l in 1:len_basis]  # (3, 3, L)
    end
    meta = Dict("elements" => [AtomsBase.atomic_number(ChemicalSpecies(el)) for el in elements],
                "order" => order, "totaldegree" => totaldegree, "wL" => WL, "rcut" => RCUT,
                "n_B" => n_B, "n_pair" => length(m.pairbasis.spec), "len_basis" => len_basis,
                "n_configs" => 3, "nnll" => nnll,
                "pair_spline" => Dict("x0" => px0, "h" => ph, "n" => pn),
                "acepotentials_version" => string(pkgversion(ACEpotentials)),
                "julia_version" => string(VERSION))
    D["meta_json"] = Vector{UInt8}(JSON.json(meta))
    out = joinpath(OUTDIR, "built_basis_ref_$(name).npz")
    npzwrite(out, D)
    @printf("wrote %s: n_B %d, len_basis %d (%.1f MB)\n", out, n_B, len_basis, filesize(out) / 2^20)
end
