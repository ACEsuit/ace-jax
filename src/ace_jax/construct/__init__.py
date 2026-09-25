"""Construct ACE basis definitions from Python (authoring path).

Scope now: the SO(3) coupling table (A2B + aa_spec), generated via an
EquivariantTensors-only JuliaCall shim -- see docs/coupling-etshim-spec.md --
and the algebraic smoothness prior (prior.py, pure numpy, parity-tested against
julia/smoothness_reference.jl).  Radials / pair basis / embedding stay exported
(separate follow-up).

prior.py is deliberately numpy-only so the eval/fit path can import it as a
fallback for exports that lack gamma (cli.py and the bench driver go through
prior.prior_diagonal, which rebuilds it from the export's meta["nnll"]); the
coupling shim is the only piece that
needs the optional `authoring` extra (juliacall + juliapkg).  Refitting on an
existing model never touches it.
"""
