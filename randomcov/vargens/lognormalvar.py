import numpy as np


def lognormal_var(n, mean=1.0, sigma=0.5):
    """
    Generates a list of random variances from a log-normal distribution.

    Parameters:
        n        (int): The number of random variances to generate.
        mean   (float): The mean of the underlying normal distribution (not the variances).
        sigma  (float): The standard deviation of the underlying normal distribution.

    Returns:
        np.ndarray: An array of random variances (positive values).
    """
    return np.random.lognormal(mean=np.log(mean), sigma=sigma, size=n)
