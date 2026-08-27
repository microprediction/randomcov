"""Seeded sweep behind the graphical-lasso table: cross-validated graphical
lasso (Friedman, Hastie and Tibshirani 2008) versus Ledoit-Wolf linear
shrinkage as precision-matrix estimators, by named ensemble at fixed (n, T).

Both estimators see the same T = 60 observations; the scoreboard is the
Frobenius loss of the estimated precision against the true precision. The
table reports the loss ratio glasso / Ledoit-Wolf, so values below one mean
the sparsity prior beat the shrinkage prior.

Run: python papers/measure_audit/glasso_sweep.py
"""
import warnings

import numpy as np
from sklearn.covariance import GraphicalLassoCV, LedoitWolf

from randomcov import CORR_GENERATORS, random_covariance_matrix

warnings.filterwarnings("ignore")
rng = np.random.default_rng(14)
n, T, reps = 30, 60, 20

rows = []
for mi, m in enumerate(CORR_GENERATORS):
    ratios = []
    for rep in range(reps):
        Sigma = np.asarray(random_covariance_matrix(n=n, corr_method=m,
                                                 rng=800000 + 1000 * mi + rep))
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(n))
        X = (L @ rng.standard_normal((n, T))).T
        Theta = np.linalg.inv(Sigma + 1e-10 * np.eye(n))
        try:
            gl = GraphicalLassoCV(max_iter=200).fit(X).precision_
        except Exception:
            continue
        lw = np.linalg.inv(LedoitWolf().fit(X).covariance_)
        ratios.append(np.linalg.norm(gl - Theta, "fro")
                      / np.linalg.norm(lw - Theta, "fro"))
    ratios = np.array(ratios)
    rows.append((m.value, float(np.median(ratios)),
                 float(np.mean(ratios < 1.0)), len(ratios)))
for name, med, frac, k in sorted(rows, key=lambda r: r[1]):
    print(f"{name:16s} median gl/LW {med:5.2f}   glasso wins {frac:5.0%}"
          f"   ({k} reps)")
