using Distributed
addprocs(31, exeflags="--project=$(Base.active_project())")
using ACEpotentials

# Training data and basis specificiation
train_xyz = "gp_iter6_sparse9k.xml.xyz"
elements = [:Si]
e_ref = Dict("Si" => -158.54496821)
energy_key= "dft_energy"
force_key = "dft_force"
virial_key = "dft_virial"
order = 4
totaldegree = [20, 20, 20, 20]

print("setting up basis...")
model = acemodel(
    elements = elements,
    Eref = e_ref,
    order = order,
    totaldegree = totaldegree,
    pair_envelope = (:r, 3),
    pair_transform = (:agnesi, 1, 4),
    envelope = (:x, 2, 3),
    transform = (:agnesi, 2, 4),
    )
@show length(model.basis) cutoff(model.basis.BB[1]);

train = JuLIP.read_extxyz(train_xyz)

#setup weights and solver
weights = Dict(
    "default" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ),
    "liq" => Dict("E" => 10.0, "F" => 0.66, "V" => 0.25 ),
    "amorph" => Dict("E" => 3.0, "F" => 0.5 , "V" => 0.1),
    "sp" => Dict("E" => 3.0, "F" => 0.5 , "V" => 0.1),
    "bc8"=> Dict("E" => 50.0, "F" => 0.5 , "V" => 0.1),
    "vacancy" => Dict("E" => 50.0, "F" => 0.5 , "V" => 0.1),
    "interstitial" => Dict("E" => 50.0, "F" => 0.5 , "V" => 0.1),
    # new
    "dia" => Dict("E" => 60.0, "F" => 2.0 , "V" => 2.0 ),
  )

solver = ACEfit.BLR(factorization=:svd)

print("fitting model...")
acefit!(model, train;
        solver = solver,
        weights = weights,
        energy_key = energy_key,
        force_key = force_key,
        virial_key = virial_key,
        export_json = "Si_4_20.json",
        smoothness = 5
        )

println("computing errors...")
err = ACEpotentials.linear_errors(train,  model;
                             energy_key = energy_key,
                             force_key = force_key,
                             virial_key = virial_key)
