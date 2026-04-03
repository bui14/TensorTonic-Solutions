import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    if seq_len < 1 or d_model < 1:
        return 0
    pos = np.arange(seq_len)[:, np.newaxis]
    
    i_even = np.arange(0, d_model, 2)
    angles_even = pos / (base ** (i_even / d_model))

    i_odd = np.arange(1, d_model, 2)
    angles_odd = pos / (base ** ((i_odd - 1) / d_model))

    pe = np.zeros((seq_len,d_model))
    pe[:, 0::2] = np.sin(angles_even)
    pe[:, 1::2] = np.cos(angles_odd)

    return pe