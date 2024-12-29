from randomcov.vargens.lognormalvar import lognormal_var

# Test with direct calling

# tests/test_random_variance_vector.py

import numpy as np
from randomcov.randomvariancevector import random_variance_vector
from randomcov.vargens.allvargens import VarMethod
import pytest

# Configure logging for tests if necessary
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_random_variance_vector_shape_enum():
    """Test if the output vector has the correct length when using Enum."""
    n = 5
    var_method = VarMethod.LOGNORMAL
    var_kwargs = {"mean": 1.0,"sigma": 1.0}
    var_vector = random_variance_vector(
        n=n,
        var_method=var_method,
        var_kwargs=var_kwargs
    )
    assert var_vector.shape == (n,), f"Expected shape ({n},), got {var_vector.shape}"


def test_random_variance_vector_shape_string():
    """Test if the output vector has the correct length when using string."""
    n = 5
    var_method = 'lognormal'
    var_kwargs = {"mean": 1.0,"sigma": 1.0}
    var_vector = random_variance_vector(
        n=n,
        var_method=var_method,
        var_kwargs=var_kwargs
    )
    assert var_vector.shape == (n,), f"Expected shape ({n},), got {var_vector.shape}"


def test_random_variance_vector_positive_values_enum():
    """Test if all variances are positive when using Enum."""
    n = 10
    var_method = VarMethod.LOGNORMAL
    var_kwargs = {"mean": 1.0,"sigma": 1.0}
    var_vector = random_variance_vector(
        n=n,
        var_method=var_method,
        var_kwargs=var_kwargs
    )
    assert np.all(var_vector > 0), "Not all variances are positive"


def test_random_variance_vector_positive_values_string():
    """Test if all variances are positive when using string."""
    n = 10
    var_method = 'lognormal'
    var_kwargs = {"mean": 1.0,"sigma": 1.0}
    var_vector = random_variance_vector(
        n=n,
        var_method=var_method,
        var_kwargs=var_kwargs
    )
    assert np.all(var_vector > 0), "Not all variances are positive"


def test_random_variance_vector_lognormal_parameters_enum():
    """Test if the lognormal variances have expected statistical properties when using Enum."""
    n = 100000
    var_method = VarMethod.LOGNORMAL
    mean = 1.0
    sigma = 1.0
    var_kwargs = {"mean": mean, "sigma": sigma}
    var_vector = random_variance_vector(
        n=n,
        var_method=var_method,
        var_kwargs=var_kwargs
    )
    # For lognormal distribution, mean of variances = exp(sigma^2)
    expected_mean = np.exp(mean + 0.5*sigma**2)
    actual_mean = np.mean(var_vector)
    assert np.isclose(actual_mean, expected_mean, atol=0.1), \
        f"Expected mean ~{expected_mean}, got {actual_mean}"


def test_random_variance_vector_lognormal_parameters_string():
    """Test if the lognormal variances have expected statistical properties when using string."""
    n = 100000
    var_method = 'lognormal'
    mean = 1.0
    sigma = 1.0
    var_kwargs = {"mean": mean, "sigma": sigma}
    var_vector = random_variance_vector(
        n=n,
        var_method=var_method,
        var_kwargs=var_kwargs
    )
    # For lognormal distribution, mean of variances = exp(sigma^2)
    expected_mean = np.exp(mean+0.5*sigma**2)
    actual_mean = np.mean(var_vector)
    assert np.isclose(actual_mean, expected_mean, atol=0.1), \
        f"Expected mean ~{expected_mean}, got {actual_mean}"


@pytest.mark.parametrize("var_method", [VarMethod.LOGNORMAL, 'lognormal'])
def test_random_variance_vector_invalid_n(var_method):
    """Test that the function raises an error for invalid n values (n <= 0)."""
    invalid_n = 0  # Invalid since n must be positive
    eta = 2.0  # Irrelevant for variance vector, but kept for consistency
    var_kwargs = {"mean": 1.0,"sigma": 1.0}
    with pytest.raises(ValueError):
        random_variance_vector(
            n=invalid_n,
            var_method=var_method,
            var_kwargs=var_kwargs
        )


@pytest.mark.parametrize("var_method", [VarMethod.LOGNORMAL, 'lognormal'])
def test_random_variance_vector_invalid_method(var_method):
    """Test that the function raises an error for unsupported variance methods."""
    n = 5
    invalid_var_method = 'unsupported_method'
    var_kwargs = {"mean": 1.0,"sigma": 1.0}
    with pytest.raises(ValueError):
        random_variance_vector(
            n=n,
            var_method=invalid_var_method,
            var_kwargs=var_kwargs
        )


@pytest.mark.parametrize("var_method", [VarMethod.LOGNORMAL, 'lognormal'])
def test_random_variance_vector_different_dimensions(var_method):
    """Test random_variance_vector function for different vector sizes."""
    for n in [1, 2, 10, 100, 1000]:
        mean = 1.0
        sigma = 1.0
        var_kwargs = {"mean": mean, "sigma": sigma}
        var_vector = random_variance_vector(
            n=n,
            var_method=var_method,
            var_kwargs=var_kwargs
        )
        assert var_vector.shape == (n,), f"Expected shape ({n},), got {var_vector.shape}"
        assert np.all(var_vector > 0), "Not all variances are positive"
        # Optional: Check statistical properties if n is large enough
        if n >= 10000:
            expected_mean = np.exp(mean+0.5*sigma**2)
            actual_mean = np.mean(var_vector)
            assert np.isclose(actual_mean, expected_mean, atol=0.1), \
                f"Expected mean ~{expected_mean}, got {actual_mean}"
