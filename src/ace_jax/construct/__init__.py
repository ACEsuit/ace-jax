"""Construct ACE basis definitions from Python (authoring path).

Scope now: the SO(3) coupling table (A2B + aa_spec), generated via an
EquivariantTensors-only JuliaCall shim -- see docs/coupling-etshim-spec.md.
Radials / pair basis / embedding stay exported (separate follow-up).

Nothing here is imported by the eval/fit path; it needs the optional `authoring`
extra (juliacall + juliapkg). Refitting on an existing model never touches it.
"""
