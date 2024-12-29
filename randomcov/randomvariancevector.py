# vargens/generate_variance_vector.py

from typing import Dict, Any, Optional, Union
from randomcov.vargens.allvargens import VAR_GENERATORS, VarMethod
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)


def random_variance_vector(
    n: int,
    var_method: Union[VarMethod, str] = VarMethod.LOGNORMAL,
    var_kwargs: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Generate a random variance vector using the specified method.

    Parameters:
    - n (int): Number of variances to generate.
    - var_method (VarMethod | str): The method to use for generating variances.
      Can be a `VarMethod` Enum member or a string representing the method name.
    - var_kwargs (dict, optional): Additional keyword arguments for the variance generator.

    Returns:
    - np.ndarray: Array of generated variances.

    Raises:
    - ValueError: If an unsupported variance method is specified or if `n` is invalid.
    """
    if var_kwargs is None:
        var_kwargs = {}

    # Validate n
    if n < 1:
        logger.error(f"Invalid value for n: {n}. n must be a positive integer.")
        raise ValueError(f"Invalid value for n: {n}. n must be a positive integer.")

    # Convert string input to VarMethod Enum if necessary
    if isinstance(var_method, str):
        try:
            var_method = VarMethod(var_method.lower())
            logger.debug(f"Converted string '{var_method}' to VarMethod Enum.")
        except ValueError:
            logger.error(f"Unsupported variance method string: '{var_method}'")
            raise ValueError(
                f"Variance generation method '{var_method}' is not supported. "
                f"Available methods are: {[method.value for method in VarMethod]}"
            )

    # Retrieve the generator function directly from VAR_GENERATORS
    generator = VAR_GENERATORS.get(var_method)
    if not generator:
        logger.error(f"Unsupported variance generation method: '{var_method}'")
        raise ValueError(
            f"Variance generation method '{var_method.value}' is not supported. "
            f"Available methods are: {[method.value for method in VarMethod]}"
        )

    logger.info(f"Generating variance vector using '{var_method.value}' method.")
    var_vector = np.array(generator(n=n, **var_kwargs))
    logger.debug(f"Generated variance vector:\n{var_vector}")
    return var_vector


if __name__ == '__main__':
    # Example usage with Enum
    try:
        var_enum = random_variance_vector(
            n=6,
            var_method=VarMethod.LOGNORMAL,
            var_kwargs={"mean": 1.0, "sigma": 0.5}
        )
        print("Variance Vector using Enum:")
        print(var_enum)
    except ValueError as e:
        logger.error(e)

    # Example usage with string
    try:
        var_str = random_variance_vector(
            n=5,
            var_method='lognormal',
            var_kwargs={"mean": 1.0, "sigma": 0.5}
        )
        print("\nVariance Vector using String:")
        print(var_str)
    except ValueError as e:
        logger.error(e)
