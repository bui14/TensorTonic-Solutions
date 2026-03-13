import numpy as np

def kernel_init(Cin: int, out_channels: int):
    n_in = 3 * 3 * Cin
    std = np.sqrt(2/n_in)
    return np.random.randn(3, 3, Cin, out_channels) * std
    
def conv3x3(x: np.ndarray, out_channels: int):
    N, H, W, Cin = x.shape
    he_normal = kernel_init(Cin, out_channels)
    output = np.zeros((N, H - 2, W - 2, out_channels))
    
    for i in range(H - 2):
        for j in range(W - 2):
            window = x[:, i:i+3, j:j+3, :]
            output[:, i, j, :] = np.tensordot(window, he_normal, axes=([1, 2, 3], [0, 1, 2]))
    return np.maximum(output,0)

def maxpooling(x: np.ndarray):
    N, H, W, Cin = x.shape
    output = np.zeros((N, H//2, W//2, Cin))

    for i in range(H//2):
        for j in range(W//2):
            window = x[:, i*2 : i*2+2, j*2 : j*2+2, :]
            output[:, i, j, :] = np.max(window, axis=(1, 2))
            
    return output
    
def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    """
    U-Net encoder block: double conv + max pool.
    """

    conv1_out = conv3x3(x, out_channels)

    skip_out = conv3x3(conv1_out, out_channels)

    pool_out = maxpooling(skip_out)

    return pool_out, skip_out