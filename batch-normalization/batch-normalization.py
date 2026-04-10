import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    if x.ndim == 2:
        axis = 0
        g, b = gamma, beta

    elif x.ndim == 4:
        axis = (0,2,3)
        g = gamma.reshape(1,-1,1,1)
        b = beta.reshape(1,-1,1,1)
    else:
        return 0
        
    mu = x.mean(axis = axis, keepdims=True)
    var = x.var(axis = axis, keepdims=True)
    
    x_hat = (x-mu) / np.sqrt(var + eps)

    out = g*x_hat + b

    return out