# Array extraction shared by the exporters (export_model.jl,
# built_basis_reference.jl): each helper reads one radial component of an
# ACEpotentials model into the npz layout the Python loader expects.
# Needs `M = ACEpotentials.Models` in scope.

# ---------------------------------------------------------------- radial spline
# SplineRnlrzzBasis evaluates  Rnl(r) = spline(x) * envelope(r, x),  x = T(r).
# The spline is Interpolations.cubic_spline_interpolation over a uniform grid on
# [-1,1]: 100 nodes, 102 B-spline coefficients (one pad each side).
function spline_arrays(basis)
    NZ = length(basis._i2z)
    s11 = basis.splines[1, 1]
    ncoef = length(s11.itp.itp.coefs)
    LEN = length(basis.spec)
    rng = s11.itp.ranges[1]
    coefs = zeros(NZ, NZ, ncoef, LEN)
    for iz = 1:NZ, jz = 1:NZ
        co = basis.splines[iz, jz].itp.itp.coefs
        for (p, c) in enumerate(co)          # p = 1..ncoef  <->  offset index 0..ncoef-1
            coefs[iz, jz, p, :] .= c
        end
        r = basis.splines[iz, jz].itp.ranges[1]
        @assert first(r) ≈ first(rng) && step(r) ≈ step(rng) && length(r) == length(rng)
    end
    return coefs, Float64(first(rng)), Float64(step(rng)), length(rng)
end

_is_spline(b) = b isa M.SplineRnlrzzBasis


# `Wnlq` stays a live parameter for the analytic branch -- training would need it
# trainable, and splines are not differentiable w.r.t. what generated them.
function analytic_arrays(basis, ps_b)
    NZ = length(basis._i2z)
    W = zeros(NZ, NZ, length(basis.spec), length(basis.polys))
    for iz = 1:NZ, jz = 1:NZ
        W[iz, jz, :, :] = ps_b.Wnlq[:, :, iz, jz]
    end
    rs = basis.polys.refstate
    return W, collect(rs.A), collect(rs.B), collect(rs.C)
end

# ---------------------------------------------------------------- transforms
# NormalizedTransform(GeneralizedAgnesiTransform):
#   y = r <= rin ? 1 : 1/(1 + a s^q/(1 + s^(q-p))),  s = (r-rin)/(r0-rin)
#   x = clamp(-1 + 2 (y - yin)/(ycut - yin), -1, 1)
function transform_params(basis)
    NZ = length(basis._i2z)
    P = zeros(NZ, NZ, 7)     # p q a rin r0 yin ycut
    for iz = 1:NZ, jz = 1:NZ
        t = basis.transforms[iz, jz]
        g = t.trans
        @assert g isa M.GeneralizedAgnesiTransform "unsupported transform $(typeof(g))"
        P[iz, jz, :] = [g.p, g.q, g.a, g.rin, g.r0, t.yin, t.ycut]
    end
    P
end

# envelopes
# many-body : PolyEnvelope2sX(x1,x2,p1,p2,s)  env = s (x-x1)^p1 (x2-x)^p2 on (x1,x2)
# pair      : ACE1_PolyEnvelope1sR(rcut,r0,p) env = s^-p - sc^-p + p sc^(-p-1)(s-sc)
function env2sx_params(basis)
    NZ = length(basis._i2z); P = zeros(NZ, NZ, 5)
    for iz = 1:NZ, jz = 1:NZ
        e = basis.envelopes[iz, jz]
        @assert e isa M.PolyEnvelope2sX "unsupported envelope $(typeof(e))"
        P[iz, jz, :] = [e.x1, e.x2, e.p1, e.p2, e.s]
    end
    P
end
# ace1_model's pair basis uses ACE1_PolyEnvelope1sR (rcut, r0, p);
# ace_model's uses PolyEnvelope1sR (rcut, p) -- a different formula, not just
# different parameters.
function env1sr_params(basis)
    NZ = length(basis._i2z)
    e1 = basis.envelopes[1, 1]
    if e1 isa M.ACE1_PolyEnvelope1sR
        P = zeros(NZ, NZ, 3)
        for iz = 1:NZ, jz = 1:NZ
            e = basis.envelopes[iz, jz]
            P[iz, jz, :] = [e.rcut, e.r0, e.p]
        end
        return P, "ace1_poly1sr"
    elseif e1 isa M.PolyEnvelope1sR
        P = zeros(NZ, NZ, 2)
        for iz = 1:NZ, jz = 1:NZ
            e = basis.envelopes[iz, jz]
            P[iz, jz, :] = [e.rcut, e.p]
        end
        return P, "poly1sr"
    end
    error("unsupported pair envelope $(typeof(e1))")
end
