import numpy as np


def get_noise(
    mechanism_type: str,
    epsilon: float = None,
    sensitivity: float = None,
    sigma: float = None,
):
    if mechanism_type == "laplace":
        return np.random.laplace(loc=0, scale=sensitivity / epsilon, size=1)
    elif mechanism_type == "geometric":
        p = 1 - np.exp(-epsilon / sensitivity)
        return (np.random.geometric(p=p, size=1) - np.random.geometric(p=p, size=1))[0]
    elif mechanism_type == "gaussian":
        return np.random.normal(loc=0, scale=sigma, size=1)[0]
    else:
        raise ValueError(
            "The mechanism type must be either laplace, geometric or gaussian"
        )
