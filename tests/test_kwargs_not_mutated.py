"""random_correlation_matrix must not write into the caller's corr_kwargs.

Before 0.2.1 it called corr_kwargs.setdefault("rng", rng), so a dict shared
across calls kept the first call's seed and every later call with a
different rng= returned the identical matrix.
"""
import numpy as np

from randomcov import random_correlation_matrix


def test_shared_kwargs_dict_is_not_mutated():
    kw = {}
    A = np.asarray(random_correlation_matrix(n=8, corr_method="ar1", corr_kwargs=kw, rng=1))
    assert kw == {}, f"caller's dict was mutated: {kw}"
    B = np.asarray(random_correlation_matrix(n=8, corr_method="ar1", corr_kwargs=kw, rng=2))
    assert not np.allclose(A, B), "second call reused the first call's seed"
