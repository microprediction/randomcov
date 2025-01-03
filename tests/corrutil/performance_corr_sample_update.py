import time
import numpy as np
from randomcov.corrgens.allcorrgens import CorrMethod
from randomcov.corrutil.corrsampleupdate import corr_sample_update
from randomcov.randomcorrelationmatrix import random_correlation_matrix
from randomcov.corrutil.corrwalk import corr_walk
from scipy.stats import qmc

def generate_qmc_samples_corr(n_samples: int, target_corr) -> "np.ndarray":
    """
    Generate quasi-Monte Carlo samples in [0,1]^dim using a Halton sequence (SciPy).
    """
    n_dim = np.shape(target_corr)[0]
    mu = np.zeros(n_dim)
    mvn_sampler = qmc.MultivariateNormalQMC(mean=mu, cov=target_corr)
    samples = mvn_sampler.random(n_samples)  # shape (n_samples, d)
    return samples



if __name__ == "__main__":
    d = 100           # dimension
    N = 1_000_000    # number of samples
    orig_corr = random_correlation_matrix(n=d, corr_method=CorrMethod.LKJ)
    target_corr = corr_walk(initial_correlation_matrix=orig_corr, steps=5, epsilon=0.01)[-1]

    # -----------------------------------------------------------
    # 1) Time how long to directly generate QMC with the correlation
    # -----------------------------------------------------------
    start_time = time.time()
    qmc_with_corr = generate_qmc_samples_corr(N, orig_corr)
    direct_qmc_time = time.time() - start_time
    print(f"Time to generate QMC samples with correlation: {direct_qmc_time:.4f} s")

    # -----------------------------------------------------------
    # 2) Time how long to generate raw QMC + run corr_sample_update
    # -----------------------------------------------------------
    # (A) Generate QMC samples (no correlation or just identity)
    start_time = time.time()
    # e.g. just generate standard normal QMC and do nothing to correlate
    # or generate raw random samples...
    raw_qmc_samples = np.random.randn(N, d)  # simpler for demonstration
    raw_generation_time = time.time() - start_time
    print(f"Time to generate raw QMC (or random) samples: {raw_generation_time:.4f} s")

    # (B) Now modify them to target_corr
    print('Applying corr_sample_update...')
    start_time = time.time()
    modded_samples = corr_sample_update(raw_qmc_samples, target_corr, orig_corr, remove_mean=False)
    mod_time = time.time() - start_time
    print(f"Time to apply corr_sample_update    : {mod_time:.4f} s")

    total_2step_time = raw_generation_time + mod_time
    print(f"Total time (raw generation + correlation fix): {total_2step_time:.4f} s")

    # Now you have direct_qmc_time (single-step QMC correlation)
    # vs. total_2step_time (generate raw + fix correlation).
