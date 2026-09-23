import numpy as np


def kernel_corr(n, d=2, length_scale=None, nu="rbf", rng=None):
    """Correlation of a Gaussian field sampled at n random points in the
    unit cube of dimension d, with length scale drawn from U(0.1, 0.5).

    nu="rbf" (the default, and the only setting the paper's audits use)
    gives the squared-exponential kernel; anything else gives Matern-3/2.
    The squared-exponential field is spatially structured and smoothly
    decaying but NOT full rank in floating point: with no nugget the
    matrix is numerically singular from n of about 100 (median rank
    fraction 0.85 at n=100, 0.27 at n=300, 0.03 at n=3000) and needs a
    positive-semidefinite floor before inversion. That degeneracy is a
    property of the kernel, and it is what makes Stein's screening effect
    fail here."""
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
