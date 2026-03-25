import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    sum = 0
    l2V = 0
    l2W = 0
    for i in range(len(v)):
        sum += v[i]*w[i]
        l2V += v[i]**2
        l2W += w[i]**2
    if l2V == 0 or l2W == 0:
        return np.nan
    cosTheta = sum / (np.sqrt(l2V)*np.sqrt(l2W))
    return np.arccos(cosTheta)
    