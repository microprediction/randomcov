"""Render the audit matrix: ensembles (rows) x audited claims (columns),
colored by whether the claim delivers under that generating measure, with
heavy outlines on the cells that match the correlation structure the claim
was demonstrated on or intended for ("home turf").

Cell values are copied from the seeded sweep outputs beside this file
(*_sweep.py); re-run those scripts to regenerate them. Scores: for ratio
audits the color is log10(median ratio) clipped to [-1, 1] (green: the
method delivers; red: it backfires); for rate audits the color is the
failure rate mapped to [0, 1] (white: calibrated; red: broken); for the
screening and n_D audits the color is the log-distortion, red side only.

Run: python papers/measure_audit/audit_matrix.py  ->  matrix.pdf, matrix.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENSEMBLES = ["sparse_precision", "ar1", "hierarchical", "block_equicorr",
             "factor", "kernel", "walk", "animals", "residuals",
             "archakov_hansen", "spectrum", "wishart", "lkj", "onion",
             "vine"]

# (label, field, kind, {ensemble: value}, home-turf ensembles)
# kinds: 'ratio' (<1 method wins), 'ratio1' (=1 claim holds, >1 fails),
#        'rate' (share of failures), 'deflate' (<1 = dimensionality lost)
AUDITS = [
    ("HRP/MV", "finance", "ratio", {
        "animals": 8.4, "ar1": 0.98, "archakov_hansen": 71.8,
        "block_equicorr": 1.09, "factor": 12.3, "hierarchical": 1.00,
        "kernel": 1.51, "lkj": 2.27, "onion": 2.83, "residuals": 5.36,
        "sparse_precision": 0.76, "spectrum": 1.11, "vine": 2.68,
        "walk": 1.23, "wishart": 1.19},
     ["hierarchical", "block_equicorr"]),
    ("MP clip", "econophys.", "ratio", {
        "animals": 1.43, "ar1": 1.20, "archakov_hansen": 1.47,
        "block_equicorr": 1.00, "factor": 1.38, "hierarchical": 0.76,
        "kernel": 1.46, "lkj": 1.22, "onion": 1.29, "residuals": 1.44,
        "sparse_precision": 0.30, "spectrum": 0.94, "vine": 1.29,
        "walk": 1.11, "wishart": 1.04},
     ["spectrum", "factor"]),
    ("screening", "geostat.", "ratio1", {
        "animals": 5.5, "ar1": 1.00, "archakov_hansen": 131.0,
        "block_equicorr": 1.04, "factor": 1.06, "hierarchical": 1.02,
        "kernel": 5.7, "lkj": 3.1, "onion": 9.0, "residuals": 4.8,
        "sparse_precision": 1.00, "spectrum": 1.32, "vine": 11.9,
        "walk": 4.8e8, "wishart": 1.50},
     ["ar1", "kernel"]),
    ("GC taper", "data assim.", "ratio", {
        "animals": 1.66, "ar1": 0.53, "archakov_hansen": 1.41,
        "block_equicorr": 1.00, "factor": 1.69, "hierarchical": 0.82,
        "kernel": 1.46, "lkj": 0.74, "onion": 0.78, "residuals": 1.71,
        "sparse_precision": 0.50, "spectrum": 0.66, "vine": 0.81,
        "walk": 1.14, "wishart": 0.66},
     ["ar1", "kernel"]),
    ("diag. load", "signal proc.", "ratio", {
        "animals": 32.8, "ar1": 0.48, "archakov_hansen": 20.5,
        "block_equicorr": 0.55, "factor": 0.94, "hierarchical": 0.45,
        "kernel": 136.0, "lkj": 0.87, "onion": 3.3, "residuals": 0.62,
        "sparse_precision": 0.48, "spectrum": 0.51, "vine": 4.1,
        "walk": 8.7e7, "wishart": 0.51},
     ["spectrum", "factor"]),
    ("glasso", "graph. models", "ratio", {
        "animals": 1.00, "ar1": 0.70, "archakov_hansen": 1.00,
        "block_equicorr": 0.78, "factor": 1.09, "hierarchical": 0.68,
        "kernel": 1.00, "lkj": 0.97, "onion": 1.00, "residuals": 1.03,
        "sparse_precision": 0.63, "spectrum": 0.99, "vine": 1.00,
        "walk": 1.00, "wishart": 0.96},
     ["sparse_precision", "ar1"]),
    ("CC target", "statistics", "ratio", {
        "animals": 1.00, "ar1": 0.92, "archakov_hansen": 1.00,
        "block_equicorr": 0.96, "factor": 1.00, "hierarchical": 0.73,
        "kernel": 0.99, "lkj": 0.99, "onion": 0.99, "residuals": 1.00,
        "sparse_precision": 0.62, "spectrum": 0.95, "vine": 1.00,
        "walk": 0.87, "wishart": 0.96},
     ["block_equicorr", "factor"]),
    ("skewers", "evol. bio.", "rate", {
        "animals": 0.10, "ar1": 1.00, "archakov_hansen": 0.00,
        "block_equicorr": 1.00, "factor": 0.00, "hierarchical": 1.00,
        "kernel": 1.00, "lkj": 1.00, "onion": 1.00, "residuals": 0.00,
        "sparse_precision": 1.00, "spectrum": 1.00, "vine": 1.00,
        "walk": 1.00, "wishart": 1.00},
     ["factor", "hierarchical"]),
    ("North rule", "climatology", "rate", {
        "animals": 0.00, "ar1": 0.20, "archakov_hansen": 0.09,
        "block_equicorr": 0.00, "factor": 0.00, "hierarchical": 0.00,
        "kernel": 0.03, "lkj": 0.22, "onion": 0.33, "residuals": 0.00,
        "sparse_precision": 1.00, "spectrum": 0.10, "vine": 0.33,
        "walk": 0.00, "wishart": 0.57},
     ["kernel", "ar1"]),
    ("Mantel test", "ecology", "rate", {
        "animals": 0.40, "ar1": 0.13, "archakov_hansen": 0.28,
        "block_equicorr": 0.15, "factor": 0.38, "hierarchical": 0.06,
        "kernel": 0.39, "lkj": 0.11, "onion": 0.05, "residuals": 0.37,
        "sparse_precision": 0.07, "spectrum": 0.09, "vine": 0.05,
        "walk": 0.08, "wishart": 0.07},
     ["kernel", "ar1"]),
    ("$n_D$ of $G$", "quant. genetics", "deflate", {
        "animals": 0.89, "ar1": 0.54, "archakov_hansen": 0.78,
        "block_equicorr": 0.77, "factor": 0.90, "hierarchical": 0.78,
        "kernel": 0.88, "lkj": 0.58, "onion": 0.57, "residuals": 0.79,
        "sparse_precision": 0.42, "spectrum": 0.52, "vine": 0.55,
        "walk": 0.83, "wishart": 0.47},
     ["factor", "hierarchical"]),
]


def score(kind, v):
    if kind == "ratio":
        return float(np.clip(np.log10(v), -1.0, 1.0))
    if kind == "ratio1":
        return float(np.clip(np.log10(v), 0.0, 1.0))
    if kind == "rate":
        return float(np.clip((v - 0.05) / 0.95, 0.0, 1.0))
    if kind == "deflate":
        return float(np.clip(-np.log10(v) / 0.4, 0.0, 1.0))
    raise ValueError(kind)


def cell_text(kind, v):
    if kind in ("ratio", "ratio1", "deflate"):
        if v >= 1e4:
            return rf"$10^{{{np.log10(v):.1f}}}$"
        if v >= 100:
            return f"{v:.0f}"
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{100 * v:.0f}\\%" if False else f"{100 * v:.0f}%"


S = np.array([[score(kind, vals[e]) for (_, _, kind, vals, _) in AUDITS]
              for e in ENSEMBLES])

fig, ax = plt.subplots(figsize=(11.5, 7.2))
im = ax.imshow(S, cmap="RdYlGn_r", vmin=-1.0, vmax=1.0, aspect="auto")

for i, e in enumerate(ENSEMBLES):
    for j, (_, _, kind, vals, home) in enumerate(AUDITS):
        v = vals[e]
        s = S[i, j]
        color = "white" if abs(s) > 0.62 else "black"
        ax.text(j, i, cell_text(kind, v), ha="center", va="center",
                fontsize=8.5, color=color)
        if e in home:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="black", linewidth=2.4))

ax.set_xticks(range(len(AUDITS)))
ax.set_xticklabels([f"{lab}\n{field}" for (lab, field, _, _, _) in AUDITS],
                   fontsize=9)
ax.set_yticks(range(len(ENSEMBLES)))
ax.set_yticklabels([e.replace("_", r"\_") if False else e
                    for e in ENSEMBLES], fontsize=9, family="monospace")
ax.set_xticks(np.arange(-0.5, len(AUDITS)), minor=True)
ax.set_yticks(np.arange(-0.5, len(ENSEMBLES)), minor=True)
ax.grid(which="minor", color="white", linewidth=1.2)
ax.tick_params(which="both", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                    ticks=[-1, 0, 1])
cbar.ax.set_yticklabels(["claim\ndelivers", "break-even /\ncalibrated",
                         "claim\nbackfires"], fontsize=8.5)
cbar.outline.set_visible(False)

ax.set_title("Eleven covariance-method claims across fifteen generating "
             "ensembles\n(outlined cells: the correlation structure the "
             "claim was demonstrated on or intended for)",
             fontsize=11, pad=12)
fig.tight_layout()
fig.savefig("papers/measure_audit/matrix.pdf")
fig.savefig("papers/measure_audit/matrix.png", dpi=170)
print("wrote matrix.pdf / matrix.png")
