
def test_gen():
    from randomcov import cov_generator
    cov = cov_generator(n=50, corr_method='residuals', var_method='lognormal')