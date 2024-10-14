from randomcov.vargens.allvargens import VAR_GENERATORS


def random_variance_vector(n, var_method:str, var_kwargs :dict=None):
    """
         Dispatch by name to variance vector generation
    """
    if var_kwargs is None:
        var_kwargs = {}

    for method in VAR_GENERATORS:
        if var_method.lower() in method.__name__.lower():
            return method(n=n, **var_kwargs)



if __name__=='__main__':
    vars = random_variance_vector(n=6, var_method='lognormal')
    print(vars)