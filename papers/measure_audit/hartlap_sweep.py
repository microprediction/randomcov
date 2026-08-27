"""Seeded null control: the Hartlap-Simon-Schneider (2007) debiasing of the
inverse sample covariance is an exact inverse-Wishart identity, so it holds
for every true covariance. An audit pipeline that finds ensemble-dependence
here has a bug.

One covariance is fixed per ensemble; 400 Gaussian panels of T observations
are drawn; the trace of the average inverse sample covariance is compared to
the trace of the true inverse, raw and corrected by (T - n - 2)/(T - 1).
The corrected ratio should be 1 under every ensemble; the raw ratio should
be the Wishart factor (T - 1)/(T - n - 2) everywhere.

Run: python papers/measure_audit/hartlap_sweep.py
"""
import warnings

import numpy as np

from randomcov import CORR_GENERATORS, random_covariance_matrix

warnings.filterwarnings("ignore")
rng = np.random.default_rng(24)
n, T, reps = 30, 60, 400
correction = (T - n - 2) / (T - 1)
print(f"Wishart factor (T-1)/(T-n-2) = {1.0 / correction:.3f}")

rows = []
for mi, m in enumerate(CORR_GENERATORS):
    Sigma = np.asarray(random_covariance_matrix(
        n=n, corr_method=m, rng=1300000 + 1000 * mi))
    Sigma = Sigma + 1e-10 * np.eye(n)
    L = np.linalg.cholesky(Sigma)
    true_tr = np.trace(np.linalg.inv(Sigma))
    trs = []
    for rep in range(reps):
        X = (L @ rng.standard_normal((n, T))).T
        trs.append(np.trace(np.linalg.inv(np.cov(X, rowvar=False))))
    raw = float(np.mean(trs) / true_tr)
    rows.append((m.value, raw, correction * raw))
for name, raw, corr in sorted(rows, key=lambda r: r[0]):
    print(f"{name:16s} raw {raw:5.2f}   corrected {corr:5.3f}")
