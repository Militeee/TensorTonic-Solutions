import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here

    pos = np.arange(seq_length)[:,np.newaxis]
    den = np.exp(
        np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
    )
  
    res = np.zeros((seq_length,d_model))

    res[:,0::2] =  np.sin(pos * den)

    res[:,1::2] =  np.cos(pos * den)

    return res
    