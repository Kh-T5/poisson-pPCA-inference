import numpy as np


def simulate_poisson_lognormal(n: int = 200, p: int = 50, q: int = 3, seed: int = 42):
    """
    Simulation de la distribution lognormal poisson. Cette fonction est appelée pour créer le dataset d'étude / d'inférence.
    """
    random = np.random.default_rng(seed)

    mu = random.normal(0, 1, p)
    B = random.normal(0, 0.5, (p, q))

    W = random.normal(0, 1, (n, q))

    Z = mu + W @ B.T

    Y = random.poisson(np.exp(Z))

    return Y, Z, W, mu, B
