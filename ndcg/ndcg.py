import math
import numpy as np

def formula(relevance_scores, k):
    relevance_scores = relevance_scores[:k]

    gains = 2**relevance_scores - 1

    discounts = np.log2(np.arange(len(relevance_scores)) + 2)
    return np.sum(gains / discounts)
    
def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    relevance_scores = np.asarray(relevance_scores)
    if k > len(relevance_scores):
        k = len(relevance_scores)

    dcg = formula(relevance_scores,k)
    print(dcg)
    idcg = formula(np.sort(relevance_scores)[::-1],k)
    print(idcg)

    if idcg == 0:
        return 0.0
    return dcg/idcg