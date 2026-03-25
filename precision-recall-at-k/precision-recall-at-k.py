import numpy as np
def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    if len(recommended) == 0 or k == 0:
        return 0
    recommended = np.asarray(recommended)
    relevant = np.asarray(relevant)
    top_k = recommended[:k]
    res = len(np.intersect1d(top_k, relevant))
    return [res/k, res/len(relevant)]