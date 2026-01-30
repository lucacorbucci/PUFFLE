import numpy as np


def get_noise(
    mechanism_type: str,
    epsilon: float | None = None,
    sensitivity: float | None = None,
    sigma: float | None = None,
) -> float:
    """Get noise from mechanism."""
    rng = np.random.default_rng()
    if mechanism_type == "laplace":
        if sensitivity is None or epsilon is None:
            msg = "Sensitivity and epsilon needed for Laplace"
            raise ValueError(msg)
        return rng.laplace(loc=0, scale=sensitivity / epsilon, size=1)
    if mechanism_type == "geometric":
        if sensitivity is None or epsilon is None:
            msg = "Sensitivity and epsilon needed for Geometric"
            raise ValueError(msg)
        p = 1 - np.exp(-epsilon / sensitivity)
        return (rng.geometric(p=p, size=1) - rng.geometric(p=p, size=1))[0]
    if mechanism_type == "gaussian":
        if sigma is None:
            msg = "Sigma needed for Gaussian"
            raise ValueError(msg)
        return rng.normal(loc=0, scale=sigma, size=1)[0]

    msg = "The mechanism type must be either laplace, geometric or gaussian"
    raise ValueError(msg)
