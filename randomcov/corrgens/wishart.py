import numpy as np
import pandas as pd
from scipy.stats import wishart

def wishart_corr(n):
    """
    Generates a correlation matrix using the Wishart distribution.

    Args:
        n (int): Number of variables.

    Returns:
        pd.DataFrame: Generated correlation matrix.
    """

    # Step 1: Generate a random positive definite scale matrix (identity for simplicity)
    scale = np.eye(n)

    # Step 2: Sample from Wishart distribution
    df = 2*n
    wishart_sample = wishart.rvs(df=df, scale=scale, size=1)

    # Step 3: Convert covariance to correlation matrix
    D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(wishart_sample)))
    corr_matrix = D_inv_sqrt @ wishart_sample @ D_inv_sqrt
    return corr_matrix


if __name__=='__main__':
    n = 5  # Number of variables
    corr_df = wishart_corr(n)
    print(corr_df)
