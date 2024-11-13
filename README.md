# randomcov
Generating random covariance and correlation matrices. 


### Install 

    pip install randomcov 

or for latest

    pip install git+https://github.com/microprediction/randomcov.git
    
### Example

    from randomcov import random_covariance_matrix
    cov = random_covariance_matrix(n=50, corr_method='residuals', var_method='lognormal')

### Motivation

To collect standard but also novel correlation and covariance generation methods, in order to better understand when some estimation methods work better than others in different contexts: such as the construction of machine learning model ensembles, combinations of forecasts, or financial portfolios.  


