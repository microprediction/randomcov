import numpy as np
from randomcov.corrutil.corrwalk import corr_walk

def walk_corr(n, rho=0.3, steps=5, epsilon=0.1, rng=None):
    corr0 = np.eye(n) * (1 - rho) + rho * np.ones((n, n))
    walk = corr_walk(initial_correlation_matrix=corr0, steps=steps,
                     epsilon=epsilon, rng=rng)
    return walk[-1]

