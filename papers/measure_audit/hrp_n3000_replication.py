"""Targeted replication of the n=3000, T/n=2 HRP-vs-long-only-minvar audit
cell for `ar1` and `hierarchical`, the two ensembles whose n=30 ratio was
close to a tie. The scale grid (scale_sweep.py) only affords twelve seeded
draws per cell at n=3000; this script appends draws twelve through
forty-seven under the identical deterministic seeding scheme
(run_cell in scale_sweep.py seeds purely from (n, ensemble_index, rep, ...),
so reps 0-11 here are bit-identical to the grid's own and reps 12-47 are a
genuine extension, not a fresh experiment).

Run: python papers/measure_audit/hrp_n3000_replication.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scale_sweep import run_cell  # noqa: E402
from randomcov import CORR_GENERATORS  # noqa: E402

N = 3000
T = 6000  # T/n = 2
REPS = 48
ENSEMBLES = ("ar1", "hierarchical")


def main():
    index = {m.value: i for i, m in enumerate(CORR_GENERATORS)}
    opts = {"audits": ["hrp"], "glasso_cv": False, "glasso_fixed": False,
            "glasso_max_iter": 100}
    tasks = [(N, index[e], e, rep, opts)
             for e in ENSEMBLES for rep in range(REPS)]

    t0 = time.time()
    values = {e: [] for e in ENSEMBLES}
    with ProcessPoolExecutor(max_workers=14) as ex:
        for fut in as_completed({ex.submit(run_cell, t): t for t in tasks}):
            task = fut.result()
            ensemble = task["ensemble"]
            v = task["values"]["hrp"].get(T)
            if v is not None:
                values[ensemble].append(v)

    summary = {}
    for e in ENSEMBLES:
        r = np.array(sorted(values[e]))
        summary[e] = {
            "reps": len(r),
            "median": float(np.median(r)),
            "min": float(r.min()),
            "max": float(r.max()),
            "frac_hrp_wins": float(np.mean(r < 1.0)),
        }
        print(f"{e}: reps={len(r)} median={np.median(r):.3f} "
              f"range=[{r.min():.3f}, {r.max():.3f}] "
              f"frac<1={np.mean(r < 1.0):.2f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "hrp_n3000_replication.json")
    with open(out_path, "w") as f:
        json.dump({"n": N, "T": T, "reps": REPS, "summary": summary}, f,
                   indent=2)
    print(f"seconds: {time.time() - t0:.1f}, written to {out_path}")


if __name__ == "__main__":
    main()
