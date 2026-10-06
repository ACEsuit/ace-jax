# D4 noise-model sensitivity: rev2 coverage table

| run | MAP σ_E / σ_F / σ_V | ID cfg-weighted (atom) | all | crack | crack tip | edge | screw | ρ crack / edge / screw | test RMSE E (meV/atom) / F (eV/Å) / V |
|---|---|---|---|---|---|---|---|---|---|
| bench365_ard_default_pol | nan / nan / nan | 0.898 (0.903) | 0.911 [0.907, 0.916] | 0.908 [0.904, 0.911] | 0.893 [0.887, 0.899] | 0.937 [0.934, 0.941] | 0.942 [0.937, 0.947] | 0.37 / 0.30 / 0.32 | 3.028 / 0.06872 / 0.7335 |
| bench365_ard_seq_pq | 0.0171 / 0.0661 / 0.102 | 0.898 (0.903) | 0.911 [0.907, 0.916] | 0.908 [0.904, 0.911] | 0.893 [0.887, 0.899] | 0.937 [0.934, 0.941] | 0.942 [0.937, 0.947] | 0.37 / 0.30 / 0.32 | 3.028 / 0.06872 / 0.7336 |
| bench365_ard_seq_shared | 0.0675 / 0.0675 / 0.0675 | 0.898 (0.904) | 0.913 [0.909, 0.917] | 0.909 [0.906, 0.913] | 0.894 [0.888, 0.899] | 0.940 [0.937, 0.943] | 0.944 [0.940, 0.949] | 0.37 / 0.28 / 0.30 | 5.415 / 0.06894 / 0.6907 |

ARD hyperparameters (h):
- bench365_ard_default_pol: log_sigma_E -4.068, log_sigma_F -2.717, log_sigma_V -2.281, a_2body -8.420, a_3body -8.420, a_4body -5.402
- bench365_ard_seq_pq: a_2body -8.403, a_3body -8.403, a_4body -5.401
- bench365_ard_seq_shared: a_2body -8.611, a_3body -8.611, a_4body -5.447
