import numpy as np


def block_equicorr(n, blocks=4, rho_within=None, rho_between=0.1,
                   rng=None):
    """Block equicorrelation (Engle & Kelly's DECO, block form):
    constant correlation within each block and a constant between
    blocks. PSD requires rho_between <= rho_within (checked)."""
    rng = np.random.default_rng(rng)
    labels = rng.integers(0, blocks, size=n)
    if rho_within is None:
        rho_within = rho_between + (0.85 - rho_between) * rng.random(blocks)
    rho_within = np.broadcast_to(np.asarray(rho_within, float), (blocks,))
    if (rho_within < rho_between).any():
        raise ValueError("need rho_within >= rho_between for PSD")
    C = np.full((n, n), rho_between)
    for b in range(blocks):
        idx = np.where(labels == b)[0]
        C[np.ix_(idx, idx)] = rho_within[b]
    np.fill_diagonal(C, 1.0)
    return C
