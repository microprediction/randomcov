"""Recompute the cross-validated glasso audit on every cell where it failed.

Finds each (ensemble, rep) at size n whose notes record a gl@T failure and
re-runs just that audit with --merge, one cell at a time (merges into the
same raw grid, so never in parallel). Other audits and cells are untouched.

    python papers/measure_audit/fill_gl_gaps.py --n 300
"""
import argparse
import json
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
args = ap.parse_args()

raw = json.load(open(os.path.join(args.outdir, f"regime_raw_n{args.n}.json")))
todo = sorted({(c["ensemble"], c["rep"]) for c in raw["cells"]
               if any(k.startswith("gl@") for k in c.get("notes", {}))})
print(f"n={args.n}: {len(todo)} cells with gl failures", flush=True)
sweep = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "scale_sweep.py")
for i, (e, rep) in enumerate(todo, 1):
    print(f"[{i}/{len(todo)}] {e} rep{rep}", flush=True)
    subprocess.run([sys.executable, sweep, "--n", str(args.n), "--only", e,
                    "--rep-start", str(rep), "--reps", "1", "--audits", "gl",
                    "--merge", "--glasso-cv", "on", "--workers", "1",
                    "--blas-threads", "1", "--outdir", args.outdir],
                   check=True)
