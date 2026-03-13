import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x_array = np.array(x)
    return x_array * (1/(1+np.exp(-x_array)))