import numpy as np
    
def conv3x3(x: np.ndarray, out_channels: int):
    N, H, W, Cin = x.shape
    output = np.zeros((N, H - 2, W - 2, out_channels))
    return output

def maxpooling(x: np.ndarray, out_channels: int):
    N, H, W, Cin = x.shape
    output = np.zeros((N, H//2, W//2, out_channels))
    return output
    
def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    """
    U-Net encoder block: double conv + max pool.
    """

    conv1_out = conv3x3(x, out_channels)

    skip_out = conv3x3(conv1_out, out_channels)

    pool_out = maxpooling(skip_out, out_channels)

    return pool_out, skip_out