"""Seeded sweep behind the screening-effect table: how much conditional
variance is lost by predicting each variable from its m = 5 most-correlated
neighbors instead of from all n - 1 others (the kriging screening effect),
by named ensemble.

This is a population property of the generating correlation matrix, so no
sampling is involved; the table reports the median over variables and draws
of condvar(m nearest) / condvar(all), which is >= 1, with 1 meaning
screening holds exactly.

Run: python papers/measure_audit/screening_sweep.py
"""
import warnings

import numpy as np

from randomcov import CORR_GENERATORS, random_correlation_matrix

warnings.filterwarnings("ignore")
n, reps, mnear = 30, 20, 5


def cond_var(C, i, J):
    J = list(J)
    return float(C[i, i] - C[i, J] @ np.linalg.solve(C[np.ix_(J, J)], C[J, i]))


rows = []
for mi, m in enumerate(CORR_GENERATORS):
    ratios = []
    for rep in range(reps):
        C = np.asarray(random_correlation_matrix(n=n, corr_method=m,
                                                  rng=400000 + 1000 * mi + rep))
        C = C + 1e-10 * np.eye(n)
        for i in range(n):
            others = [j for j in range(n) if j != i]
            near = sorted(others, key=lambda j: -abs(C[i, j]))[:mnear]
            full = cond_var(C, i, others)
            ratios.append(cond_var(C, i, near) / max(full, 1e-12))
    ratios = np.array(ratios)
    rows.append((m.value, float(np.median(ratios)),
                 float(np.mean(ratios < 1.1))))
for name, med, frac in sorted(rows, key=lambda r: r[1]):
    print(f"{name:16s} median near/full {med:8.2f}   within 10% {frac:5.0%}")
