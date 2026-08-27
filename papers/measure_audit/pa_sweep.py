"""Seeded sweep behind the factor-retention table: number of components
retained by Horn's (1965) parallel analysis (95th-percentile criterion) and
by the Kaiser (1960) eigenvalue-greater-than-one rule, by named ensemble at
fixed (n, T). The Kaiser column also reports the population count of
eigenvalues above one, so over- and under-extraction are visible.

Run: python papers/measure_audit/pa_sweep.py
"""
import warnings

import numpy as np

from randomcov import CORR_GENERATORS, random_correlation_matrix

warnings.filterwarnings("ignore")
rng = np.random.default_rng(8)
n, T, reps, nsim = 30, 60, 20, 200


def corr_eigs(X):
    S = np.cov(X, rowvar=False)
    s = np.sqrt(np.diag(S))
    return np.sort(np.linalg.eigvalsh(S / np.outer(s, s)))[::-1]


# null eigenvalue quantiles from independent normal data of the same shape
null = np.array([corr_eigs(rng.standard_normal((T, n))) for _ in range(nsim)])
q95 = np.quantile(null, 0.95, axis=0)

rows = []
for mi, m in enumerate(CORR_GENERATORS):
    counts, kaiser, ktrue = [], [], []
    for rep in range(reps):
        C = np.asarray(random_correlation_matrix(n=n, corr_method=m,
                                                  rng=500000 + 1000 * mi + rep))
        L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
        X = (L @ rng.standard_normal((n, T))).T
        eigs = corr_eigs(X)
        counts.append(int(np.sum(eigs > q95)))
        kaiser.append(int(np.sum(eigs > 1.0)))
        ktrue.append(int(np.sum(np.linalg.eigvalsh(C) > 1.0)))
    rows.append((m.value, float(np.mean(counts)), float(np.std(counts)),
                 float(np.mean(kaiser)), float(np.mean(ktrue))))
for name, mu, sd, kz, kt in sorted(rows, key=lambda r: r[1]):
    print(f"{name:16s} PA {mu:5.1f} ({sd:.1f})   kaiser {kz:5.1f}"
          f"  true>1 {kt:5.1f}")
