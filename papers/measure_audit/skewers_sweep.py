"""Seeded sweep behind the random-skewers table: false-similarity rate of
Cheverud's (1996) random-skewers comparison of covariance matrices, by
named ensemble.

Two INDEPENDENT draws from the same ensemble are compared with 1000 common
random skewers; the pair is declared "similar" when the mean response-vector
correlation exceeds the 95th percentile of the standard null (correlation
between independent random vectors). Since the matrices are independent, a
sound test should declare similarity ~5% of the time; the table reports the
mean skewers correlation and the realized false-similarity rate, a
population property of the generating measure (no sampling noise involved).

Run: python papers/measure_audit/skewers_sweep.py
"""
import warnings

import numpy as np

from randomcov import CORR_GENERATORS, random_correlation_matrix

warnings.filterwarnings("ignore")
rng = np.random.default_rng(16)
n, npairs, nskew = 30, 20, 1000

# null: mean correlation between responses is compared against the
# distribution of the correlation of independent random vectors
null = rng.standard_normal((20000, 2, n))
null_corr = np.array([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                      for a, b in null])
q95 = np.quantile(null_corr, 0.95)

skewers = rng.standard_normal((nskew, n))
skewers /= np.linalg.norm(skewers, axis=1, keepdims=True)

rows = []
for mi, m in enumerate(CORR_GENERATORS):
    means, declared = [], []
    for pair in range(npairs):
        C1 = np.asarray(random_correlation_matrix(
            n=n, corr_method=m, rng=900000 + 1000 * mi + 2 * pair))
        C2 = np.asarray(random_correlation_matrix(
            n=n, corr_method=m, rng=900000 + 1000 * mi + 2 * pair + 1))
        r1, r2 = skewers @ C1, skewers @ C2
        cs = np.sum(r1 * r2, axis=1) / (np.linalg.norm(r1, axis=1)
                                        * np.linalg.norm(r2, axis=1))
        means.append(float(cs.mean()))
        declared.append(cs.mean() > q95)
    rows.append((m.value, float(np.mean(means)), float(np.mean(declared))))
print(f"null 95th percentile: {q95:.3f}")
for name, mu, rate in sorted(rows, key=lambda r: r[2]):
    print(f"{name:16s} mean skewers corr {mu:5.2f}   "
          f"false similarity {rate:4.0%}")
