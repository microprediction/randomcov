# randomcov
Generating random covariance and correlation matrices


### Install 

    pip install randomcov 

or for latest

    pip install git+https://github.com/microprediction/randomcov.git
    

### Example

    from randomcov import random_covariance_matrix
    cov = random_covariance_matrix(n=50, corr_method='residuals', var_method='lognormal')
