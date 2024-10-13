from randomcov.corrgenerator import corr_generator
from randomcov.vargenerator import var_generator
import math
import numpy as np

def safe_sqrt(x):
    try:
        return math.sqrt(x)
    except ArithmeticError:
        return 0


def cov_generator(n, corr_method, var_method, corr_kwargs:dict=None, var_kwargs:dict=None):
    """

         Sample

    :param n:
    :param corr_method:
    :param var_method:
    :param corr_kwargs:
    :param cov_kwargs:
    :return:
    """
    corr = np.array(corr_generator(n=n, corr_method=corr_method, corr_kwargs=corr_kwargs))
    vars = var_generator(n=n, var_method=var_method, var_kwargs=var_kwargs)
    devos = np.array([ math.sqrt(v) for v in vars ])
    cov = corr * devos[:, np.newaxis] * devos[np.newaxis, :]
    return cov



if __name__=='__main__':
    cov = cov_generator(n=5, corr_method='lkj', var_method='lognormal')
    print(cov)