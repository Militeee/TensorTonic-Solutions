import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    # Your code here
    mu = np.mean(x, axis = -1, keepdims=True)
    sigma = np.var(x, axis = -1, keepdims=True)

    centered = x - mu

    eps = 1e-6

    scaled = centered / np.sqrt(sigma + eps)

    return (gamma * scaled) + beta