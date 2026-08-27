"""Seeded sweep behind the leaderboard matrix: the covariance estimators
of scikit-learn and of precise (the batch-applicable ones), ranked per
ensemble at fixed (n, T).

Each estimator sees the same twenty seeded panels per ensemble; the score
is the median relative Frobenius loss ||est - Sigma|| / ||Sigma|| against
the generating covariance, and the deliverable is the rank of each
estimator within each ensemble (1 = best of the seventeen). Results are written
to ranking_results.json for the figure script ranking_matrix.py.

Run: python papers/measure_audit/ranking_sweep.py
"""
import json
import warnings

import numpy as np
import precise
from sklearn.covariance import (OAS, EmpiricalCovariance, GraphicalLassoCV,
                                LedoitWolf, MinCovDet, ShrunkCovariance)

from randomcov import CORR_GENERATORS, random_covariance_matrix

warnings.filterwarnings("ignore")
rng = np.random.default_rng(26)
n, T, reps = 30, 60, 20

SKLEARN = [
    ("Empirical", EmpiricalCovariance),
    ("LedoitWolf", LedoitWolf),
    ("OAS", OAS),
    ("Shrunk", ShrunkCovariance),
    ("GraphLassoCV", GraphicalLassoCV),
    ("MinCovDet", MinCovDet),
]
PRECISE = ["AdaptiveEwaCovariance", "ConditionalCovariance", "DCCCovariance",
           "DiagonalCovariance", "EwaCovariance", "FactorCovariance",
           "GeodesicEwaCovariance", "HuberCovariance",
           "PartialMomentsCovariance", "SchurCovariance", "TylerCovariance"]


def fit_precise(name, X):
    est = getattr(precise, name)()
    for row in X:
        est.partial_fit(row)
    return est.covariance_


NAMES = ([f"skl:{name}" for name, _ in SKLEARN]
         + [f"pre:{name.replace('Covariance', '')}" for name in PRECISE])

results = {}
for mi, m in enumerate(CORR_GENERATORS):
    losses = {name: [] for name in NAMES}
    for rep in range(reps):
        Sigma = np.asarray(random_covariance_matrix(
            n=n, corr_method=m, rng=1400000 + 1000 * mi + rep))
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(n))
        X = (L @ rng.standard_normal((n, T))).T
        denom = np.linalg.norm(Sigma, "fro")
        for label, cls in zip(NAMES, [c for _, c in SKLEARN] + PRECISE):
            try:
                if label.startswith("skl:"):
                    est = cls().fit(X).covariance_
                else:
                    est = fit_precise(cls, X)
                losses[label].append(np.linalg.norm(est - Sigma, "fro")
                                     / denom)
            except Exception:
                losses[label].append(np.inf)
    med = {name: float(np.median(v)) for name, v in losses.items()}
    order = sorted(med, key=med.get)
    results[m.value] = {"loss": med,
                        "rank": {name: order.index(name) + 1
                                 for name in med}}

with open("papers/measure_audit/ranking_results.json", "w") as f:
    json.dump(results, f, indent=1)

print("\nwinner per ensemble:")
for e, r in results.items():
    best = min(r["loss"], key=r["loss"].get)
    print(f"  {e:16s} {best}")
