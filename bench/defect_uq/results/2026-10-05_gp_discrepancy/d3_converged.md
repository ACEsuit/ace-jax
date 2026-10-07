# D3 residual-GP prior inside / outside the inducing span

gp_model `acegp-data/results/2026-10-06-gp-discrepancy-d4/bench365_gp_conv/gp_model.npz` (theta, inducing set and posterior as saved); clusters 2 r_cut + 0.5 A.

| atoms | n | median k (eV^2) | median (k-q)/k | median k_F (eV^2/A^2) | median (k_F-q_F)/k_F [p10, p90] | median s (A) |
|---|---|---|---|---|---|---|
| tip r<5 | 6 | 1.921e-03 | 0.002 | 2.500e-02 | 0.005 [0.003, 0.006] | 2.122 |
| crack 10-20 | 6 | 4.515e-03 | 0.002 | 4.389e-02 | 0.004 [0.002, 0.006] | 2.081 |
| edge core r<5 | 10 | 4.215e-03 | 0.001 | 2.706e-02 | 0.005 [0.004, 0.007] | 2.084 |
| screw core r<5 | 10 | 6.409e-03 | 0.002 | 4.652e-02 | 0.004 [0.003, 0.006] | 2.063 |
| big-cell bulk r 18-26 | 6 | 3.736e-03 | 0.002 | 2.443e-02 | 0.004 [0.002, 0.008] | 2.092 |
| bench365 test (ID) | 10 | 3.581e-03 | 0.003 | 3.050e-02 | 0.005 [0.002, 0.008] | 2.096 |

Predictive force variance (SoR posterior + derivative DTC), summed over the target's 3 components:

| atoms | median SoR posterior var (eV^2/A^2) | median DTC var | median DTC / predictive [p10, p90] |
|---|---|---|---|
| tip r<5 | 6.450e-04 | 1.175e-04 | 0.138 [0.107, 0.238] |
| crack 10-20 | 4.758e-04 | 1.548e-04 | 0.207 [0.158, 0.299] |
| edge core r<5 | 5.174e-04 | 1.464e-04 | 0.214 [0.156, 0.271] |
| screw core r<5 | 5.010e-04 | 1.710e-04 | 0.244 [0.157, 0.372] |
| big-cell bulk r 18-26 | 5.556e-04 | 1.542e-04 | 0.180 [0.124, 0.239] |
| bench365 test (ID) | 2.265e-04 | 1.261e-04 | 0.343 [0.253, 0.519] |
