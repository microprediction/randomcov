# tests/test_random_correlation_matrix.py

import numpy as np
from numpy.linalg import eigvals
from randomcov.randomcorrelationmatrix import random_correlation_matrix
from randomcov.corrgens.allcorrgens import CorrMethod
import pytest

# Configure logging for tests
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_random_correlation_matrix_shape_enum():
    """Test if the output matrix has the correct shape when using Enum."""
    n = 5
    eta = 2.0
    corr_matrix = random_correlation_matrix(
        n=n,
        corr_method=CorrMethod.LKJ,
        corr_kwargs={"eta": eta}
    )
    assert corr_matrix.shape == (n, n), f"Expected shape ({n}, {n}), got {corr_matrix.shape}"


def test_random_correlation_matrix_shape_string():
    """Test if the output matrix has the correct shape when using string."""
    n = 5
    eta = 2.0
    corr_matrix = random_correlation_matrix(
        n=n,
        corr_method='lkj',
        corr_kwargs={"eta": eta}
    )
    assert corr_matrix.shape == (n, n), f"Expected shape ({n}, {n}), got {corr_matrix.shape}"


def test_random_correlation_matrix_symmetry_enum():
    """Test if the output matrix is symmetric when using Enum."""
    n = 5
    eta = 2.0
    corr_matrix = random_correlation_matrix(
        n=n,
        corr_method=CorrMethod.LKJ,
        corr_kwargs={"eta": eta}
    )
    assert np.allclose(corr_matrix, corr_matrix.T), "Matrix is not symmetric"


def test_random_correlation_matrix_symmetry_string():
    """Test if the output matrix is symmetric when using string."""
    n = 5
    eta = 2.0
    corr_matrix = random_correlation_matrix(
        n=n,
        corr_method='lkj',
        corr_kwargs={"eta": eta}
    )
    assert np.allclose(corr_matrix, corr_matrix.T), "Matrix is not symmetric"


def test_random_correlation_matrix_positive_definite_enum():
    """Test if the output matrix is positive definite when using Enum."""
    n = 5
    eta = 2.0
    corr_matrix = random_correlation_matrix(
        n=n,
        corr_method=CorrMethod.LKJ,
        corr_kwargs={"eta": eta}
    )
    eigenvalues = eigvals(corr_matrix)
    assert np.all(eigenvalues > 0), "Matrix is not positive definite"


def test_random_correlation_matrix_positive_definite_string():
    """Test if the output matrix is positive definite when using string."""
    n = 5
    eta = 2.0
    corr_matrix = random_correlation_matrix(
        n=n,
        corr_method='lkj',
        corr_kwargs={"eta": eta}
    )
    eigenvalues = eigvals(corr_matrix)
    assert np.all(eigenvalues > 0), "Matrix is not positive definite"

