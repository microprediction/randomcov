# randomdov.corrutil.corrsampleupdate
import numpy as np
import time

def corr_sample_update(samples: np.ndarray,
                       target_corr: np.ndarray,
                       original_corr: np.ndarray = None,
                       remove_mean: bool = True,
                       regularization: float = 1e-12) -> np.ndarray:
    """
    Transform `samples` so that they match a provided target correlation matrix, with added stability.

    Assumptions:
      - `samples` is an (N, d) array, where N is # of samples, d is dimension.
      - Each column of `samples` has approximately unit variance (and mean ~ 0, if remove_mean=False).
      - `target_corr` is a (d, d) desired correlation matrix (positive-definite).

    Steps:
      1) (Optional) Remove empirical mean from `samples`.
      2) Compute the sample correlation of the existing `samples`.
      3) Add a small diagonal ridge (regularization) to mitigate near-singularity.
      4) Factor old sample correlation (Cholesky).
      5) Factor target_corr (Cholesky).
      6) Solve for T such that T = new_sqrt @ (old_sqrt)^(-1) in a stable manner.
      7) Apply T to each sample.
      8) (Optional) Re-add mean if desired.
         (Typically not done if we want zero-mean final samples for correlation.)

    :param samples: (N x d) array, zero-mean & unit-variance columns recommended
    :param target_corr: (d x d) desired correlation matrix (must be pos-def)
    :param remove_mean: Whether to subtract the empirical mean of each column before processing
    :param regularization: Small positive float added to the diagonal of the old correlation
                           to improve numerical stability
    :return: (N x d) new samples whose empirical correlation approximates `target_corr`.
    """
    # ---------------------------------------------------------------------
    # 1) (Optional) Remove empirical mean
    # ---------------------------------------------------------------------
    import time
    st = time.time()
    st0 = st

    if remove_mean:
        col_means = np.mean(samples, axis=0)
        samples_centered = samples - col_means
    else:
        col_means = np.zeros(samples.shape[1])
        samples_centered = samples

    N, d = samples_centered.shape
    print(f'  ... step 1 {time.time()-st} s')

    # ---------------------------------------------------------------------
    # 2) Compute the old sample correlation
    # ---------------------------------------------------------------------


    if original_corr is None:
        # Sample covariance:
        st = time.time()
        sample_cov = np.cov(samples_centered, rowvar=False)  # shape (d, d)
        print(f'step 2 (cov) {time.time()-st}')

        # Convert to correlation by dividing out sqrt of diagonal
        st = time.time()
        stds = np.sqrt(np.diag(sample_cov))
        # Avoid zeros in the diagonal
        stds = np.where(stds < 1e-15, 1e-15, stds)
        sample_corr = sample_cov / np.outer(stds, stds)
        print(f'  ... step 2 {time.time()-st}')
    else:
        sample_corr = original_corr

    # ---------------------------------------------------------------------
    # 3) Regularize the old correlation (ridge)
    # ---------------------------------------------------------------------
    # Add epsilon on the diagonal to help with near-singularity
    st = time.time()
    sample_corr_reg = sample_corr + regularization * np.eye(d)
    print(f'eye time {time.time()-st} s')

    # ---------------------------------------------------------------------
    # 4) Factor old sample correlation (Cholesky)
    # ---------------------------------------------------------------------
    st = time.time()
    try:
        old_sqrt = np.linalg.cholesky(sample_corr_reg)
    except np.linalg.LinAlgError as e:
        raise ValueError("Old sample correlation (regularized) not positive-definite.") from e
    print(f'... step 4 (chol)  {time.time()-st} s')

    # ---------------------------------------------------------------------
    # 5) Factor target_corr (Cholesky)
    # ---------------------------------------------------------------------
    st = time.time()
    try:
        new_sqrt = np.linalg.cholesky(target_corr)
    except np.linalg.LinAlgError as e:
        raise ValueError("Target correlation not positive-definite.") from e
    print(f'... step 5 (chol)  {time.time() - st} s')

    # ---------------------------------------------------------------------
    # 6) Solve for T = new_sqrt @ inv(old_sqrt) in a stable manner
    #    We want T * old_sqrt = new_sqrt => T = new_sqrt * old_sqrt^-1
    #    Instead of computing old_sqrt^-1 directly, use np.linalg.solve on transposes.
    # ---------------------------------------------------------------------
    # T * old_sqrt = new_sqrt
    # => T^T = old_sqrt^T \ new_sqrt^T  (where "\" denotes solve)
    # => T = ( old_sqrt^T \ new_sqrt^T )^T
    st = time.time()
    temp = np.linalg.solve(old_sqrt.T, new_sqrt.T)  # shape (d, d)
    T = temp.T  # shape (d, d)
    print(f'... step 6 (solve) {time.time()-st} s')

    # ---------------------------------------------------------------------
    # 7) Apply T to each sample => new_samples = samples_centered @ T^T
    #    Because if x' = T x, and x is row-vector => x' = x T
    #    So for array of shape (N, d), we do: new_samples = samples_centered @ T^T
    # ---------------------------------------------------------------------
    import time
    st = time.time()
    new_samples_centered = samples_centered @ T.T
    print(f'... step 7 (mult) {time.time()-st}s')

    # ---------------------------------------------------------------------
    # 8) (Optional) Re-add mean (often 0 if we truly want correlation-only)
    # ---------------------------------------------------------------------
    # Typically for correlation we might keep zero-mean.
    # If needed, re-add the old means or some new means:

    print(f' ... total inside time {time.time()-st0}')
    return new_samples_centered
