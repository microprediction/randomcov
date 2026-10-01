#!/bin/zsh
# Resume n=1000 after the Sep 26-27 power loss. Sequential: every step
# merges into regime_raw_n1000.json.
set -e
cd /Users/petercotton/github/randomcov
PY=.venv/bin/python
D=papers/measure_audit
COMMON=(--n 1000 --merge --glasso-cv on --workers 3 --blas-threads 1 --outdir $D)
$PY $D/scale_sweep.py $COMMON --only vine --include-slow --rep-start 6 --reps 6
$PY $D/scale_sweep.py $COMMON --reps 12 \
  --only archakov_hansen,spectrum,factor,hierarchical,block_equicorr,ar1,kernel,sparse_precision
$PY $D/fill_gl_gaps.py --n 1000 --outdir $D
echo CHAIN DONE
