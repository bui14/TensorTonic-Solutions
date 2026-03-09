import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    if np.sum(p) != 1:
        raise ValueError
    if len(x) != len(p):
        raise ValueError
    result = np.dot(x,p)
    return result
