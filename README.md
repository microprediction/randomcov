# randomcov
Generating random covariance and correlation matrices with beautiful geodesic interpolation visualizations! 🎯


### Install 

    pip install randomcov 

or for latest

    pip install git+https://github.com/microprediction/randomcov.git
    
### Example

    from randomcov import random_covariance_matrix
    cov = random_covariance_matrix(n=50, corr_method='residuals', var_method='lognormal')


### Motivation

To collect standard but also novel correlation and covariance generation methods, in order to better understand when some estimation methods work better than others in different contexts: such as the construction of machine learning model ensembles, combinations of forecasts, or financial portfolios.  

The geodesic interpolation capabilities enable smooth transformations between covariance structures while preserving mathematical properties, making it ideal for portfolio optimization and risk management applications.


## Correlation Inflation

An example of the kind of thing I wish to test against generative models. 

**[Correlation Inflation: A Working Paper](https://github.com/microprediction/home/blob/main/workingpapers/CorrelationInflation.pdf)**

*The transformation preserves geometric properties while smoothly interpolating towards perfect correlation structure using differential geometry on the manifold of positive definite matrices.*

![Geodesic Interpolation Towards Perfect Correlation](https://github.com/microprediction/randomcov/blob/main/geodesic_interpolation_beautiful.gif)


