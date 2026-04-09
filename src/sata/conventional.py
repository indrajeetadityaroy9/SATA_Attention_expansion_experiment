import math
import torch


def conventional_attention(Q, K, V, is_causal=False):
    """Standard O(n^2) softmax attention baseline. Returns [..., n_qry, d_val]."""
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
    if is_causal:
        n = scores.size(-1)
        scores = scores + torch.triu(scores.new_full((n, n), float('-inf')), diagonal=1)
    return torch.softmax(scores, dim=-1) @ V
