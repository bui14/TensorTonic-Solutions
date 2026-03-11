import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    if y is None or len(y) == 0:
        return 0.0
    result = 0
    values, counts = np.unique(y, return_counts=True)
    sum = np.sum(counts)
    for i in counts:
        p = i/sum
        result -= p* np.log2(p)
    return float(result)