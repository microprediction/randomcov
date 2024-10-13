from randomcov.vargens.allvargens import VAR_GENERATORS


def var_generator(n, var_method:str, var_kwargs :dict=None):
    """
         Dispatch to variance generation
    """
    if var_kwargs is None:
        var_kwargs = {}

    for method in VAR_GENERATORS:
        if var_method.lower() in method.__name__.lower():
            return method(n=n, **var_kwargs)



if __name__=='__main__':
    vars = var_generator(n=6, var_method='lognormal')
    print(vars)