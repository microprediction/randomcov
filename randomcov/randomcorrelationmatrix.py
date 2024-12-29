# randomcov/randomcorrelationmatrix.py

from typing import Dict, Any, Optional, Union
from randomcov.corrgens.allcorrgens import CORR_GENERATORS, CorrMethod
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)


def random_correlation_matrix(
    n: int,
    corr_method: Union[CorrMethod, str] = CorrMethod.LKJ,
    corr_kwargs: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Generate a random correlation matrix using the specified method.

    Parameters:
    - n (int): The size of the correlation matrix.
    - corr_method (CorrMethod | str): The method to use for generating correlations.
      Can be a `CorrMethod` Enum member or a string representing the method name.
    - corr_kwargs (dict, optional): Additional keyword arguments for the correlation generator.

    Returns:
    - np.ndarray: The generated correlation matrix.

    Raises:
    - ValueError: If an unsupported correlation method is specified.
    """
    logger.info("Starting correlation matrix generation.")

    if corr_kwargs is None:
        corr_kwargs = {}

    # Convert string input to CorrMethod Enum if necessary
    if isinstance(corr_method, str):
        try:
            corr_method = CorrMethod(corr_method.lower())
            logger.debug(f"Converted string '{corr_method}' to CorrMethod Enum.")
        except ValueError:
            logger.error(f"Unsupported correlation method string: '{corr_method}'")
            raise ValueError(
                f"Correlation method '{corr_method}' is not supported. "
                f"Available methods are: {[method.value for method in CorrMethod]}"
            )

    # Retrieve the generator function directly from CORR_GENERATORS
    generator = CORR_GENERATORS.get(corr_method)
    if not generator:
        logger.error(f"Unsupported correlation method: '{corr_method}'")
        raise ValueError(
            f"Correlation method '{corr_method}' is not supported. "
            f"Available methods are: {[method.value for method in CorrMethod]}"
        )

    logger.info(f"Generating correlation matrix using '{corr_method.value}' method.")
    correlation_matrix = np.array(generator(n=n, **corr_kwargs))
    logger.debug(f"Generated correlation matrix:\n{correlation_matrix}")
    return correlation_matrix


if __name__ == '__main__':
    # Example usage with Enum
    try:
        corr_enum = random_correlation_matrix(
            n=6,
            corr_method=CorrMethod.LKJ,
            corr_kwargs={"eta": 10.0}
        )
        print("Correlation Matrix using Enum:")
        print(corr_enum)
    except ValueError as e:
        logger.error(e)

    # Example usage with string
    try:
        corr_str = random_correlation_matrix(
            n=5,
            corr_method='lkj',
            corr_kwargs={"eta": 10.0}
        )
        print("\nCorrelation Matrix using String:")
        print(corr_str)
    except ValueError as e:
        logger.error(e)
