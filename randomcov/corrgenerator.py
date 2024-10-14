from randomcov.corrgens.allcorrgens import CORR_GENERATORS


def corr_generator(n, corr_method:str, corr_kwargs :dict=None):
    """
         Dispatch to correlation generation
    """
    if corr_kwargs is None:
        corr_kwargs = {}

    for method in CORR_GENERATORS:
        if corr_method.lower() in method.__name__.lower():
            return method(n=n, **corr_kwargs)



if __name__=='__main__':
    corr = corr_generator(n=6, corr_method='lkj', corr_kwargs={"eta":10.0})
    print(corr)