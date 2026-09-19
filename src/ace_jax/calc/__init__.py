def __getattr__(name):
    if name == "ACECalculator":
        from .point import ACECalculator
        return ACECalculator
    if name == "GPCalculator":
        from .gp import GPCalculator, FittedGP, fit_posteriors
        return {"GPCalculator": GPCalculator, "FittedGP": FittedGP,
                "fit_posteriors": fit_posteriors}[name]
    raise AttributeError(name)
