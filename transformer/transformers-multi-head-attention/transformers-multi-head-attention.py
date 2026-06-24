import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here

    B, S, D = Q.shape

    Q_p = np.dot(Q, W_q).reshape(B, S, num_heads, D//num_heads).transpose(0,2,1,3)
    K_p = np.dot(K, W_k).reshape(B, S, num_heads, D//num_heads).transpose(0,2,3,1)
    V_p = np.dot(V, W_v).reshape(B, S, num_heads, D//num_heads).transpose(0,2,1,3)

    scores = (np.matmul(Q_p, K_p) / np.sqrt(D//num_heads))
    A = softmax(scores) 

    out = np.matmul(A, V_p)

    return np.matmul(out.transpose(0,2,1,3).reshape(B, S,D), W_o)

    

    