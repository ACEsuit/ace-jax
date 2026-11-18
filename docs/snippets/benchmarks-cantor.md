| Code and model | A100 GPU | CPU node (128 cores) |
|---|--:|--:|
| ace-jax, linear ACE, kernel (C++ on the CPU, CUDA on the GPU) | 0.15 | 0.376 |
| ace-jax, linear ACE, compiled JAX program | 0.591 | 3.85 |
| ace-jax, PACE model, compiled JAX program | 0.718 | 5.61 |
| ML-PACE, the same PACE model | 0.296 | 0.505 |
| MACE-MP-0b2 medium, Symmetrix | 7.54 | 29.4 |
