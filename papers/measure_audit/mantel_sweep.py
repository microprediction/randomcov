"""Seeded sweep behind the Mantel table: realized type-I error of the
Mantel (1967) permutation test between distance matrices of two INDEPENDENT
variables observed at n correlated sites, by named ensemble.

Guillot and Rousset (2013) reported inflation of 25-55% under spatially
autocorrelated (Matern-type) fields; the audit measures the inflation as a
function of the generating measure. Each rep draws a site correlation C,
two independent site variables x, y ~ N(0, C), forms Euclidean distance
matrices, and runs the Mantel test with 499 label permutations at nominal
5%. The table reports the rejection rate over 100 reps.

Run: python papers/measure_audit/mantel_sweep.py
"""
import warnings

import numpy as np

from randomcov import CORR_GENERATORS, random_correlation_matrix

warnings.filterwarnings("ignore")
rng = np.random.default_rng(22)
n, reps, nperm = 30, 100, 499
iu = np.triu_indices(n, 1)


def mantel_p(x, y, rng):
    Dx = np.abs(x[:, None] - x[None, :])[iu]
    Dy_full = np.abs(y[:, None] - y[None, :])
    obs = np.corrcoef(Dx, Dy_full[iu])[0, 1]
    count = 0
    for _ in range(nperm):
        p = rng.permutation(n)
        if np.corrcoef(Dx, Dy_full[np.ix_(p, p)][iu])[0, 1] >= obs:
            count += 1
    return (1 + count) / (1 + nperm)


rows = []
for mi, m in enumerate(CORR_GENERATORS):
    rej = 0
    for rep in range(reps):
        C = np.asarray(random_correlation_matrix(
            n=n, corr_method=m, rng=1200000 + 1000 * mi + rep))
        L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
        x, y = (L @ rng.standard_normal((n, 2))).T
        if mantel_p(x, y, rng) <= 0.05:
            rej += 1
    rows.append((m.value, rej / reps))
for name, rate in sorted(rows, key=lambda r: r[1]):
    print(f"{name:16s} type-I error {rate:4.0%}   (nominal 5%)")
