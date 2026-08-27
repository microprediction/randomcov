import numpy as np


def _givens_fix(A, tol=1e-12):
    """Davies & Higham (2000) / Bendel & Mickey (1978): rotate a
    symmetric PSD matrix with trace n to unit diagonal by Givens
    rotations, preserving the spectrum exactly."""
    n = len(A)
    for _ in range(2 * n * n):
        d = np.diag(A)
        if np.abs(d - 1.0).max() < tol:
            break
        i = int(np.argmin(d))
        j = int(np.argmax(d))
        if d[i] > 1.0 - tol or d[j] < 1.0 + tol:
            break
        aij = A[i, j]
        disc = aij * aij - (d[i] - 1.0) * (d[j] - 1.0)
        t = (aij + np.sign(aij if aij != 0 else 1.0) * np.sqrt(max(disc, 0.0))) \
            / (d[j] - 1.0)
        c = 1.0 / np.sqrt(1.0 + t * t)
        s = c * t
        G = np.eye(n)
        G[i, i] = c; G[j, j] = c
        G[i, j] = s; G[j, i] = -s
        A = G.T @ A @ G
        A = (A + A.T) / 2.0
    np.fill_diagonal(A, 1.0)
    return A


def _spectrum(n, kind, rng, q=2.0, spikes=3, spike_size=None):
    if kind == "exp":                     # Bendel-Mickey's classic choice
        lam = rng.exponential(size=n)
    elif kind == "dirichlet":
        lam = rng.dirichlet(np.ones(n) * 2.0)
    elif kind == "marchenko_pastur":      # Wishart bulk, aspect ratio q=T/n
        lam = np.sort(np.linalg.eigvalsh(np.corrcoef(
            rng.normal(size=(n, max(int(q * n), n + 1))))))
    elif kind == "spiked":                # Johnstone: bulk + a few spikes
        lam = rng.exponential(size=n) * 0.5
        k = min(spikes, n)
        top = spike_size if spike_size is not None else n / (2.0 * k)
        lam[:k] = top * (1.0 + rng.random(k))
    else:
        raise ValueError(f"unknown spectrum kind {kind}")
    lam = np.maximum(lam, 1e-12)
    return lam * (n / lam.sum())


def spectrum_corr(n, kind="dirichlet", rng=None, **kw):
    """Correlation matrix with a PRESCRIBED random spectrum: draw
    eigenvalues (exp = Bendel-Mickey 1978; dirichlet; marchenko_pastur;
    spiked = Johnstone 2001), rotate by a Haar frame, then restore the
    unit diagonal exactly with Givens rotations (Davies-Higham 2000),
    which preserves the eigenvalues."""
    rng = np.random.default_rng(rng)
    lam = _spectrum(n, kind, rng, **kw)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    A = (Q * lam) @ Q.T
    A = (A + A.T) / 2.0
    return _givens_fix(A)
