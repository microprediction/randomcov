import numpy as np
import pandas as pd
from scipy.stats import wishart

def wishart_corr(n, rng=None):
    """
    Normalized Wishart sample correlation: the sample correlation of 2n
    independent standard Gaussian observations of n variables, so the
    aspect ratio T/n is fixed at 2 and the eigenvalue bulk follows the
    Marchenko-Pastur law for that ratio. This is the estimation-noise
    ensemble, with no population structure at all.

    Args:
        n (int): Number of variables. Degrees of freedom are 2n.

    Returns:
        np.ndarray: Generated correlation matrix.
    """

    # Step 1: Generate a random positive definite scale matrix (identity for simplicity)
    scale = np.eye(n)

    # Step 2: Sample from Wishart distribution
    df = 2*n
    wishart_sample = wishart.rvs(df=df, scale=scale, size=1,
                                 random_state=np.random.default_rng(rng))

    # Step 3: Convert covariance to correlation matrix
    D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(wishart_sample)))
    corr_matrix = D_inv_sqrt @ wishart_sample @ D_inv_sqrt
    return corr_matrix


if __name__=='__main__':
    n = 5  # Number of variables
    corr_df = wishart_corr(n)
    print(corr_df)
