import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.asarray(x, dtype=np.float64)
    
    if rng is None:
        rand = np.random.random(x.shape)
    else:
        rand = rng.random(x.shape)

    mask = rand > p
    scale = 1.0 / (1.0-p)
    dropout_pattern = mask.astype(x.dtype) * scale

    train_arr = x * dropout_pattern
    
    return train_arr, dropout_pattern