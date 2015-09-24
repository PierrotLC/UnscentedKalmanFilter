
'''
KF utilities
'''

import numpy as np


def _determine_dimensionality(variables, default):
    '''Derive state space dimensionality

    Params
    ------
    variables : list of ({None, array}, conversion function, index)
        variables, functions to convert them to arrays, and indices in those
        arrays to derive dimensionality from.
    default : {None, int}
        default dimensionality to return if variables is empty

    Returns
    -------
    dim : int
        dimensionality of state space as derived from variables or default.
    '''
    
    # gather possible values based on the variables
    candidates = []
    for (v, converter, idx) in variables:
        if v is not None:
            v = converter(v)
            candidates.append(v.shape[idx])

    # also use the manually specified default
    if default is not None:
        candidates.append(default)

    # ensure consistency of all derived values
    if len(candidates) == 0:
        return 1
    else:
        if not np.all(np.array(candidates) == candidates[0]):
            raise ValueError(
                "The shape of all " +
                "parameters is not consistent.  " +
                "Please re-check their values."
            )
        return candidates[0]


def _last_dims(X, t, ndims=2):
    '''X final dimensions extraction

    Extract the final ndim dimensions at index t if X has >= ndim + 1
    dimensions, otherwise return X.

    Params
    ------
    X : array with at least ndims dimension
    t : int
        index to use for the ndims + 1th dimension
    ndims : optional int
        dimensions desired number in X

    Returns
    -------
    Y : array of ndims dimension
        the final ndims dimensions indexed by t
    '''
    
    X = np.asarray(X)
    if len(X.shape) == ndims + 1:
        return X[t]
    elif len(X.shape) == ndims:
        return X
    else:
        raise ValueError(("X only has %d dimensions when %d" +
                " or more are required") % (len(X.shape), ndims))


def _arg_or_default(arg, default, dim, name):
    if arg is None:
        result = np.asarray(default)
    else:
        result = arg
    if len(result.shape) > dim:
        raise ValueError(
            ('%s is not constant for all time.'
             + '  You must specify it manually.') % (name,)
        )
    return result
