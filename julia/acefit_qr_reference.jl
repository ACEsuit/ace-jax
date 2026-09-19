# ACEfit QR reference for the linear (M = 0) equivalence test.
#
#   julia --project=acejax/julia acejax/julia/acefit_qr_reference.jl
#
# Reads the exported ACEfit design matrix (fixtures/si_tiny_design.npz: the raw
# rows A, targets Y, per-row weights W, smoothness prior gamma) and solves the
# regularised weighted least squares with ACEfit's OWN solver -- the same
# qr([sqrt(W).*A; (1/sigma_c) diag(gamma)]) \ [sqrt(W).*Y; 0] that acefit! runs.
# Writes fixtures/si_tiny_acefit_qr.npz {C, sigma_c} as the cross-language
# reference for tests/test_gp_solve.py::test_matches_acefit_qr_reference.
using ACEfit, NPZ, LinearAlgebra

const FIX = joinpath(@__DIR__, "..", "fixtures")
d = npzread(joinpath(FIX, "si_tiny_design.npz"))
A, Y, W, gamma = d["A"], d["Y"], d["W"], d["gamma"]
sigma_c = 0.5
sw = sqrt.(W)
Aw, yw = sw .* A, sw .* Y
C = ACEfit.solve(ACEfit.QR(lambda = 1 / sigma_c, P = Diagonal(gamma)), Aw, yw)["C"]
npzwrite(joinpath(FIX, "si_tiny_acefit_qr.npz"), Dict("C" => C, "sigma_c" => [sigma_c]))
println("wrote si_tiny_acefit_qr.npz  (P = ", length(C), ", sigma_c = ", sigma_c, ")")
