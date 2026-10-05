!!! warning "Enable float64 in Python"
    Fitting and radial learning need float64. The `aj` command enables
    float64 itself. In Python, enable float64 before any other code imports
    JAX:

    ```python
    import jax
    jax.config.update("jax_enable_x64", True)
    ```

    Alternatively, set `JAX_ENABLE_X64=1` in the environment. ace-jax never
    changes the JAX precision setting itself. The JAX default is float32.
