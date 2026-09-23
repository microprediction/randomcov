import numpy as np


def block_equicorr(n, blocks=4, rho_within=None, rho_between=0.1,
                   rng=None):
    """Block equicorrelation (Engle & Kelly's DECO, block form):
    constant correlation within each block and a constant between blocks.

    Positive semidefiniteness is checked exactly. rho_within >= rho_between
    is necessary but not sufficient: with rho_between negative the guard
    alone admits matrices with eigenvalues below -3 at the defaults. The
    spectrum of a block-equicorrelation matrix is (1 - rho_within[b]) with
    multiplicity m_b - 1 for each block of size m_b, plus the eigenvalues of
    the B x B reduced matrix R with R_bb = 1 + (m_b - 1) rho_within[b] and
    R_bc = rho_between sqrt(m_b m_c). The matrix is PSD iff both parts are.
    """
    rng = np.random.default_rng(rng)
    labels = rng.integers(0, blocks, size=n)
    if rho_within is None:
        rho_within = rho_between + (0.85 - rho_between) * rng.random(blocks)
    rho_within = np.broadcast_to(np.asarray(rho_within, float), (blocks,))
    if (rho_within < rho_between).any():
        raise ValueError("need rho_within >= rho_between for PSD")
    sizes = np.bincount(labels, minlength=blocks)
    live = sizes > 0
    if (rho_within[live] > 1.0).any():
        raise ValueError("rho_within must not exceed 1")
    m = sizes[live].astype(float)
    R = rho_between * np.sqrt(np.outer(m, m))
    np.fill_diagonal(R, 1.0 + (m - 1.0) * rho_within[live])
    if np.linalg.eigvalsh(R).min() < -1e-10:
        raise ValueError(
            "block equicorrelation is not positive semidefinite for these "
            "parameters: with rho_between < 0 the between-block matrix needs "
            "rho_between >= -min_b rho_within[b] / (blocks - 1), roughly")
    C = np.full((n, n), rho_between)
    for b in range(blocks):
        idx = np.where(labels == b)[0]
        C[np.ix_(idx, idx)] = rho_within[b]
    np.fill_diagonal(C, 1.0)
    return C
