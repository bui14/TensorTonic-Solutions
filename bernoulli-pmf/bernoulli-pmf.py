import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    x = np.asarray(x)
    pmf = np.where(x == 0, 1-p, p)
    return pmf, p, p*(1-p)