import numpy as np
import pytest

from randomcov.corrutil.isvalidcorr import is_valid_corr


def test_valid():
    assert is_valid_corr(np.array([[1.0, 0.5], [0.5, 1.0]]))


def test_indefinite():
    assert not is_valid_corr(np.array([[1.0, 2.0], [2.0, 1.0]]))


@pytest.mark.parametrize("x", [np.nan, np.inf, -np.inf])
def test_non_finite_off_diagonal(x):
    assert not is_valid_corr(np.array([[1.0, x], [x, 1.0]]))


def test_non_square():
    assert not is_valid_corr(np.ones((2, 3)))


def test_non_2d():
    assert not is_valid_corr(np.ones(3))


def test_non_numeric():
    assert not is_valid_corr([["a", "b"], ["c", "d"]])
