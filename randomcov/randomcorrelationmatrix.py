from randomcov.corrgens.allcorrgens import CORR_GENERATORS
import numpy as np


def random_correlation_matrix(n, corr_method='lkj', corr_kwargs :dict=None):
    """
         Dispatch to correlation generation
    """
    if corr_kwargs is None:
        corr_kwargs = {}

    for gen in CORR_GENERATORS:
        if corr_method.lower() in gen.__name__.lower():
            return np.array(gen(n=n, **corr_kwargs))



if __name__=='__main__':
    corr = random_correlation_matrix(n=6, corr_method='lkj', corr_kwargs={"eta":10.0})
    print(corr)