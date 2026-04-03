import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    x = np.asarray(x)
    q = np.asarray(q)
    np.sort(x)
    return np.percentile(x, q)