"""GRACE-1L for the ablations: tensorpotential 0.6.2.1's GRACE_1LAYER_latest preset
(potentials/presets.py::GRACE_1LAYER_v2_25), copied with one switch, `readout`:

- "mlp" (the preset): FunctionReduceN to n_mlp_dens + 1 densities, LinMLPOut2ScalarTarget (one linear
  density plus a [64] silu MLP over the other n_mlp_dens);
- "linear": FunctionReduceN to one density, added to the energy as it is (LinearOut2Target), so the site
  energy is linear in the (learned) product basis, as in linear ACE.

Frozen radials need no change here: fit_grace passes fit::trainable_variable_names without the
radial MLP's variables (`R/`), so it stays at its random initialisation, a fixed radial basis.
gracemaker imports this module by name (input.yaml potential::custom: grace_custom.grace_1l) from the
run directory, where fit_grace copies it; the trained model.yaml holds the instructions, so loading
and export never need it.  Without TensorFlow in ace-jax's environment, nothing else imports it."""
from tensorpotential import constants
from tensorpotential.instructions import (BondLength, ConstantScaleShiftTarget, CreateOutputTarget, FCRight2Left,
                                          FunctionReduceN, InstructionManager, LinearOut2Target, LinMLPOut2ScalarTarget,
                                          MLPRadialFunction_v2, ProductFunction, RadialBasis, ScalarChemicalEmbedding,
                                          ScaledBondVector, SingleParticleBasisFunctionScalarInd, SphericalHarmonic,
                                          TrainableShiftTarget)
from tensorpotential.utils import Parity


def grace_1l(element_map, rcut=6, avg_n_neigh=1.0, constant_out_shift=0.0, constant_out_scale=1.0, lmax=4,
             basis_type="Cheb", cutoff_function_order=16, n_rad_base=10, n_rad_max=32, prod_func_n_max=64,
             embedding_size=128, n_mlp_dens=16, max_order=4, func_init="random", chem_init="random",
             cutoff_dict=None, atomic_shift_map=None, readout="mlp", dense_nbr=False, **kwargs):
    if cutoff_dict is not None:
        raise ValueError("grace_custom.grace_1l: one cutoff only")
    if readout not in ("mlp", "linear"):
        raise ValueError(f"readout {readout!r}: 'mlp' or 'linear'")
    num_elements = len(element_map)
    with InstructionManager(dense_nbr=dense_nbr) as instructor:
        d_ij = BondLength()
        rhat = ScaledBondVector(bond_length=d_ij)
        g_k = RadialBasis(bonds=d_ij, basis_type=basis_type, nfunc=n_rad_base, p=cutoff_function_order,
                          normalized=False, rcut=rcut)
        z = ScalarChemicalEmbedding(element_map=element_map, embedding_size=embedding_size, name="Z", init=chem_init)
        R_nl = MLPRadialFunction_v2(n_rad_max=n_rad_max, lmax=lmax, basis=g_k, name="R", hidden_layers=[64, 64],
                                    activation=["silu", "silu"])
        Y = SphericalHarmonic(vhat=rhat, lmax=lmax, name="Y")
        A = SingleParticleBasisFunctionScalarInd(radial=R_nl, angular=Y, indicator=z, name="A", avg_n_neigh=avg_n_neigh)
        instructions = [A]
        if max_order > 1:
            A1 = FCRight2Left(left=A, right=A, name="A1", n_out=prod_func_n_max, norm_out=True)
            AA = ProductFunction(left=A1, right=A1, name="AA", lmax=lmax, Lmax=lmax, keep_parity=Parity.REAL_PARITY,
                                 is_left_right_equal=True, normalize=True)
            instructions.append(AA)
        if max_order > 2:
            AA1 = FCRight2Left(left=AA, right=A, name="AA1", n_out=prod_func_n_max, norm_out=True)
            AAA = ProductFunction(left=AA1, right=A1, name="AAA", lmax=lmax, Lmax=0, keep_parity=Parity.REAL_PARITY,
                                  normalize=True)
            instructions.append(AAA)
        if max_order > 3:
            AA2 = FCRight2Left(left=AA, right=A, name="AA2", n_out=prod_func_n_max, norm_out=True)
            AAAA = ProductFunction(left=AA2, right=AA2, name="AAAA", lmax=lmax, Lmax=0,
                                   keep_parity=Parity.REAL_PARITY, normalize=True)
            instructions.append(AAAA)
        n_out = n_mlp_dens + 1 if readout == "mlp" else 1
        instr_red = FunctionReduceN(instructions=instructions, name="rho", ls_max=0, n_out=n_out,
                                    is_central_atom_type_dependent=True, number_of_atom_types=num_elements,
                                    allowed_l_p=Parity.REAL_PARITY, init_vars=func_init)
        out_instr = CreateOutputTarget(name=constants.PREDICT_ATOMIC_ENERGY)
        if readout == "mlp":
            LinMLPOut2ScalarTarget(origin=[instr_red], target=out_instr, hidden_layers=[64], activation="silu")
        else:
            LinearOut2Target(origin=[instr_red], target=out_instr)
        if constant_out_shift != 0 or constant_out_scale != 1 or atomic_shift_map is not None:
            ConstantScaleShiftTarget(target=out_instr, scale=constant_out_scale, shift=constant_out_shift,
                                     atomic_shift_map=atomic_shift_map)
        TrainableShiftTarget(target=out_instr, number_of_atom_types=num_elements)
    return instructor
