import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    for i in range(len(x)):
        if x[i] < 0:
            x[i]*= alpha

    return np.array(x, dtype=float)