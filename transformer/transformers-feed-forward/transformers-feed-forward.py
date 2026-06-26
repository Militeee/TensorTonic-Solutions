import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here
    z1 = np.dot(x, W1) + b1
    a1 = np.maximum(0, z1)

    return np.dot(a1, W2) + b2