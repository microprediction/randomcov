"""Seeded sweep behind the shrinkage-intensity table: Ledoit-Wolf
intensity by named ensemble at fixed (n, T).

Run: python papers/measure_audit/lw_sweep.py
"""
import warnings

import numpy as np
from randomcov import CORR_GENERATORS, random_correlation_matrix
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1)
n, T, reps = 30, 60, 20
rows = []
for mi, m in enumerate(CORR_GENERATORS):
    vals = []
    for rep in range(reps):
        C = np.asarray(random_correlation_matrix(n=n, corr_method=m,
                                                  rng=0 + 1000 * mi + rep))
        L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
        X = (L @ rng.standard_normal((n, T))).T
        vals.append(LedoitWolf().fit(X).shrinkage_)
    rows.append((m.value, float(np.mean(vals)), float(np.std(vals))))
for name, mu, sd in sorted(rows, key=lambda r: r[1]):
    print(f"{name:16s} {mu:.3f} ({sd:.3f})")
