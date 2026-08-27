import numpy as np


def hierarchical_corr(n, rho_top=0.1, rho_gain=None, rng=None):
    """Ultrametric (cophenetic) correlation from a random dendrogram:
    random binary merges with correlations increasing toward the leaves
    (Tumminello, Lillo & Mantegna's hierarchically nested factor world;
    exactly the class HRP believes in, and exactly a tree race)."""
    rng = np.random.default_rng(rng)
    nodes = [[i] for i in range(n)]
    levels = []
    while len(nodes) > 1:
        i, j = sorted(rng.choice(len(nodes), 2, replace=False), reverse=True)
        merged = nodes[i] + nodes[j]
        levels.append(merged)
        del nodes[i]; del nodes[j]
        nodes.append(merged)
    C = np.full((n, n), rho_top)
    # later merges are higher in the tree; walk from root down, raising
    # the correlation of each merged group
    rho = rho_top
    for members in reversed(levels):
        gain = rho_gain if rho_gain is not None else 0.6 * rng.random()
        rho_here = rho_top + (1.0 - rho_top) * (1.0 - np.exp(-gain * (
            n / max(len(members), 1)) / n * 3.0))
        idx = np.array(members)
        cur = C[np.ix_(idx, idx)]
        C[np.ix_(idx, idx)] = np.maximum(cur, rho_here)
    np.fill_diagonal(C, 1.0)
    # ultrametric max-of-mins structure is PSD by construction when built
    # from nested sets; guard numerically anyway
    w_, U_ = np.linalg.eigh((C + C.T) / 2.0)
    C = (U_ * np.maximum(w_, 1e-9)) @ U_.T
    d = np.sqrt(np.diag(C)); C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C
