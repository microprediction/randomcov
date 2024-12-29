# randomcov/random_covariance_matrix.py

from typing import Dict, Any, Optional, Union
from randomcov.corrgens.allcorrgens import CorrMethod
from randomcov.vargens.allvargens import VarMethod
from randomcov.randomcorrelationmatrix import  random_correlation_matrix
from randomcov.randomvariancevector import random_variance_vector
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Dispatcher that accepts ENUM or string


def random_covariance_matrix(
        n: int,
        corr_method: Union[CorrMethod, str] = CorrMethod.LKJ,
        var_method: Union[VarMethod, str] = VarMethod.LOGNORMAL,
        corr_kwargs: Optional[Dict[str, Any]] = None,
        var_kwargs: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Generate a random covariance matrix using specified correlation and variance methods.

    Parameters:
    - n (int): Dimension of the covariance matrix.
    - corr_method (CorrMethod | str): Method for generating the correlation matrix.
    - var_method (VarMethod | str): Method for generating the variance vector.
    - corr_kwargs (dict, optional): Additional keyword arguments for the correlation generator.
    - var_kwargs (dict, optional): Additional keyword arguments for the variance generator.

    Returns:
    - np.ndarray: The generated covariance matrix.
    """

    def safe_sqrt(x: float) -> float:
        try:
            return math.sqrt(x)
        except (ArithmeticError, ValueError):
            logger.warning(f"Cannot compute sqrt of {x}. Returning 0.")
            return 0.0

    logger.info("Starting covariance matrix generation.")

    # Convert string inputs to Enums if necessary
    if isinstance(corr_method, str):
        try:
            corr_method = CorrMethod(corr_method.lower())
        except ValueError:
            logger.error(f"Unsupported correlation method string: '{corr_method}'")
            raise ValueError(
                f"Correlation method '{corr_method}' is not supported. "
                f"Available methods are: {[method.value for method in CorrMethod]}"
            )

    if isinstance(var_method, str):
        try:
            var_method = VarMethod(var_method.lower())
        except ValueError:
            logger.error(f"Unsupported variance method string: '{var_method}'")
            raise ValueError(
                f"Variance method '{var_method}' is not supported. "
                f"Available methods are: {[method.value for method in VarMethod]}"
            )

    # Generate correlation matrix
    corr = random_correlation_matrix(
        n=n,
        corr_method=corr_method,
        corr_kwargs=corr_kwargs
    )

    # Generate variance vector
    vars_vector = random_variance_vector(
        n=n,
        var_method=var_method,
        var_kwargs=var_kwargs
    )

    # Compute standard deviations (devos)
    devos = np.array([safe_sqrt(v) for v in vars_vector])

    # Compute covariance matrix using outer product for efficiency
    cov = corr * np.outer(devos, devos)

    logger.info("Covariance matrix generation completed.")
    return cov
