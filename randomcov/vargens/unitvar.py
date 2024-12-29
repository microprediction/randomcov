import numpy as np


def unit_var(n):
    """
    Generates a list of unit variances.

    Parameters:
        n (int): The number of variances to generate.

    Returns:
        np.ndarray: An array of variances, all equal to 1.0.
    """
    if n < 1:
        raise ValueError("The number of variances 'n' must be a positive integer.")

    return np.ones(n)