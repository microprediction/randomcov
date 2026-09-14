"""Targeted replication of the n=3000, T/n=2 MP-clipping audit cell for
`spectrum`, `archakov_hansen`, and `kernel`, the three literature
ensembles whose 12-draw grid median sat within 2% of the clip/raw=1
flip threshold (0.996, 1.020, 1.017). The HRP check
(hrp_n3000_replication.py) found a 12-rep median 26% off 1 that did not
replicate; these clip cells are far closer to the threshold and so, if
anything, more exposed to the same risk. Same deterministic seeding as
the grid: reps 0-11 are bit-identical to scale_sweep.py's own, reps
12-47 extend it.

Run: python papers/measure_audit/clip_n3000_replication.py
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
ENSEMBLES = ("spectrum", "archakov_hansen", "kernel")


def main():
    index = {m.value: i for i, m in enumerate(CORR_GENERATORS)}
    opts = {"audits": ["clip"], "glasso_cv": False, "glasso_fixed": False,
            "glasso_max_iter": 100}
    tasks = [(N, index[e], e, rep, opts)
             for e in ENSEMBLES for rep in range(REPS)]

    t0 = time.time()
    values = {e: [] for e in ENSEMBLES}
    with ProcessPoolExecutor(max_workers=16) as ex:
        for fut in as_completed({ex.submit(run_cell, t): t for t in tasks}):
            task = fut.result()
            ensemble = task["ensemble"]
            v = task["values"]["clip"].get(T)
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
            "frac_below_1": float(np.mean(r < 1.0)),
        }
        print(f"{e}: reps={len(r)} median={np.median(r):.4f} "
              f"range=[{r.min():.4f}, {r.max():.4f}] "
              f"frac<1={np.mean(r < 1.0):.2f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "clip_n3000_replication.json")
    with open(out_path, "w") as f:
        json.dump({"n": N, "T": T, "reps": REPS, "summary": summary}, f,
                   indent=2)
    print(f"seconds: {time.time() - t0:.1f}, written to {out_path}")


if __name__ == "__main__":
    main()
