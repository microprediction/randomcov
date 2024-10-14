from randomcov.randomcorrelationmatrix import random_correlation_matrix
from randomcov.randomvariancevector import random_variance_vector
import math
import numpy as np


#  A covariance generator is usually a combination of a correlation generator and a variance generator





def random_covariance_matrix(n, corr_method:str='lkj', var_method:str='lognormal', corr_kwargs:dict=None, var_kwargs:dict=None):
    """

        Returns a random covariance matrix

    :param n:                 Dimension
    :param corr_method:       Name of correlation matrix generation method
    :param var_method:        Name of variance vector generation method
    :param corr_kwargs:       Optional keyword arguments for correlation matrix generation method
    :param var_kwargs:        Optional keyword arguments for variance vector generation method
    :return:
    """

    def safe_sqrt(x):
        try:
            return math.sqrt(x)
        except ArithmeticError:
            return 0

    corr = random_correlation_matrix(n=n, corr_method=corr_method, corr_kwargs=corr_kwargs)
    vars = random_variance_vector(n=n, var_method=var_method, var_kwargs=var_kwargs)
    devos = np.array([ safe_sqrt(v) for v in vars ])
    cov = corr * devos[:, np.newaxis] * devos[np.newaxis, :]
    return cov



if __name__=='__main__':
    print(random_covariance_matrix(n=6))
    cov = random_covariance_matrix(n=5, corr_method='lkj', var_method='lognormal')
    print(cov)

