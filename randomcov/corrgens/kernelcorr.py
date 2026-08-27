import numpy as np


def kernel_corr(n, d=2, length_scale=None, nu="rbf", rng=None):
    """Correlation of a Gaussian field sampled at n random points: RBF
    or Matern-3/2 kernel on a random point cloud in d dimensions --
    spatially structured, smoothly decaying, full rank."""
    rng = np.random.default_rng(rng)
    X = rng.random((n, d))
    if length_scale is None:
        length_scale = 0.1 + 0.4 * rng.random()
    r = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2) / length_scale
    if nu == "rbf":
        C = np.exp(-0.5 * r * r)
    else:                                  # matern 3/2
        C = (1.0 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    np.fill_diagonal(C, 1.0)
    return C
