import math
import torch


class ConventionalAttention(torch.nn.Module):
    """
    Conventional softmax attention baseline for comparison.

    Args:
        d_key: number of elements per head in queries and keys.
        is_causal: if True, applies causal (lower-triangular) mask.

    Inputs:
        Q: float tensor of shape [..., n_qry, d_key].
        K: float tensor of shape [..., n_tok, d_key].
        V: float tensor of shape [..., n_tok, d_val].

    Output:
        Y: float tensor of shape [..., n_qry, d_val].
    """

    def __init__(self, d_key: int, is_causal: bool = False) -> None:
        super().__init__()
        self.d_key = d_key
        self.is_causal = is_causal

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        scale = math.sqrt(self.d_key)
        scores = Q @ K.transpose(-2, -1) / scale                             # [..., n_qry, n_tok]

        if self.is_causal:
            n = scores.size(-1)
            mask = torch.triu(
                torch.full((n, n), float('-inf'), device=scores.device, dtype=scores.dtype),
                diagonal=1,
            )
            scores = scores + mask

        weights = torch.softmax(scores, dim=-1)                               # [..., n_qry, n_tok]
        return weights @ V                                                    # [..., n_qry, d_val]
