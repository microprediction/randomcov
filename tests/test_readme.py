
def test_gen():
    from randomcov import random_covariance_matrix
    cov = random_covariance_matrix(n=50, corr_method='residuals', var_method='lognormal')