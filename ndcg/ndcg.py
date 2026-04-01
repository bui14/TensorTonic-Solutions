import math
import numpy as np

def formula(relevance_scores, k):
    res = 0
    for i in range(k):
        gain = (2**relevance_scores[i] - 1)
        dis = (np.log2(i+2))
        res +=  gain / dis
    return res
    
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