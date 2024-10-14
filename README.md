# randomcov
Generating random covariance and correlation matrices


### Install 

    pip install randomcov 

### Example

    from randomcov import cov_generator
    cov = cov_generator(n=50, corr_method='residuals', var_method='lognormal')