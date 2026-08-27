"""Seeded sweep behind the effective-dimensions table: Kirkpatrick's (2009)
n_D = sum(lambda) / lambda_max for genetic covariance matrices, true versus
estimated, by named ensemble at fixed (n, T).

T = 15 reflects the low effective precision of typical breeding-design G
estimates. Published surveys report n_D of roughly 1-2 for estimated G
matrices and read this as biology; the audit asks how much of the number
is estimation.
Each draw provides a true covariance and a sample covariance from T
observations; the table reports the median true n_D, the median estimated
n_D, and their ratio (below one: the estimate deflates dimensionality).

Run: python papers/measure_audit/gdim_sweep.py
"""
import warnings

import numpy as np

from randomcov import CORR_GENERATORS, random_covariance_matrix

warnings.filterwarnings("ignore")
rng = np.random.default_rng(20)
n, T, reps = 30, 15, 20


def n_dim(M):
    lam = np.linalg.eigvalsh(M)
    return float(lam.sum() / lam.max())


rows = []
for mi, m in enumerate(CORR_GENERATORS):
    true_nd, hat_nd = [], []
    for rep in range(reps):
        Sigma = np.asarray(random_covariance_matrix(
            n=n, corr_method=m, rng=1100000 + 1000 * mi + rep))
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(n))
        X = (L @ rng.standard_normal((n, T))).T
        true_nd.append(n_dim(Sigma))
        hat_nd.append(n_dim(np.cov(X, rowvar=False)))
    t, h = np.median(true_nd), np.median(hat_nd)
    rows.append((m.value, float(t), float(h), float(h / t)))
for name, t, h, r in sorted(rows, key=lambda x: x[3]):
    print(f"{name:16s} true n_D {t:5.1f}   est n_D {h:5.1f}   ratio {r:4.2f}")
