import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    if not A:
        return []

    rows=len(A)
    cols=len(A[0])
    result = [[0 for _ in range(rows)] for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            result[j][i]= A[i][j]
    
    result = np.array(result)
    return result
