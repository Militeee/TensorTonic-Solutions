import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mu = np.mean(x, axis = -1, keepdims = True)
    var = np.var(x, axis = -1, keepdims = True)

    frac = (x - mu) / np.sqrt(var + eps)

    return (gamma * frac) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """

    batch, seq, emb_dim = Q.shape
    
    
    # Your code here
    Q_p, K_p, V_p = Q @ W_q, K @ W_k, V @ W_v

    Q_p = Q_p.reshape(batch, seq, num_heads, emb_dim // num_heads)
    K_p = K_p.reshape(batch, seq, num_heads, emb_dim // num_heads)
    V_p = V_p.reshape(batch, seq, num_heads, emb_dim // num_heads)

    att = softmax(Q_p.transpose(0,2,1,3) @ K_p.transpose(0,2,3,1) / np.sqrt(emb_dim // num_heads))

    out = (att @ V_p.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3).reshape(batch, seq, emb_dim)


    return out @ W_o

    

    
    
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    a1 = np.maximum(0, x @ W1 + b1)
    return a1 @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    x_new = layer_norm(x + multi_head_attention(x, x, x, 
                                                W_q, W_k, W_v, 
                                                W_o, num_heads),
                                               gamma1, beta1, 1e-6)

    out = layer_norm(x_new + feed_forward(x_new, W1, b1, W2, b2), 
                    gamma2, beta2, 1e-6)

    return out