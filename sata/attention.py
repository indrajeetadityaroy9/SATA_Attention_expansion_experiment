import math
import torch

from .taylor_term import TightlyPackedTaylorTerm


class SymmetryAwareTaylorApproximatedAttention(torch.nn.Module):
    """
    Self-attention at constant cost per token via symmetry-aware Taylor
    approximation (Heinsen and Kozachkov, 2026).

    Args:
        d_key: number of elements per head in queries and keys.
        d_val: number of elements per head in values.
        is_causal: if True, computes autoregressive attention.
        n_taylor: number of Taylor terms (controls precision).

    Inputs:
        Q: float tensor of shape [..., n_qry, d_key].
        K: float tensor of shape [..., n_tok, d_key].
        V: float tensor of shape [..., n_tok, d_val].
        continue_prev: bool, if True, continues the sequence.

    Output:
        Y: float tensor of shape [..., n_qry, d_val] with attention.
    """

    def __init__(self, d_key: int, d_val: int, is_causal: bool, n_taylor: int = 4) -> None:
        super().__init__()
        self.d_key, self.d_val, self.is_causal, self.n_taylor = (d_key, d_val, is_causal, n_taylor)
        self.tptts = torch.nn.ModuleList([
            TightlyPackedTaylorTerm(d_key, d_val, p, is_causal) for p in range(n_taylor)
        ])

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        continue_prev: bool = False,
    ) -> torch.Tensor:
        iter_over_tptts = (tptt(Q, K, V, continue_prev) for tptt in self.tptts)
        S_terms, Z_terms = zip(*iter_over_tptts)
        S = torch.stack(S_terms, dim=0).sum(dim=0)
        Z = torch.stack(Z_terms, dim=0).sum(dim=0)
        Y = torch.nan_to_num(S / Z)
        return Y

    def reset_state(self) -> None:
        """Clear accumulated states for starting a new sequence."""
        for tptt in self.tptts:
            if hasattr(tptt, 'prev_H_S'):
                del tptt.prev_H_S
            if hasattr(tptt, 'prev_H_Z'):
                del tptt.prev_H_Z

    # Convenience methods:

    def get_hidden_state_sizes(self) -> dict:
        szs = {
            f'tptt[{i}]': math.comb(tptt.d_key + tptt.p - 1, tptt.p) * (tptt.d_val + 1)
            for i, tptt in enumerate(self.tptts)
        }
        szs['Total'] = sum(v for k, v in szs.items())
        return szs

    def get_forward_FLOPs_per_query_head(self) -> dict:
        fpt = {
            f'tptt[{i}]': (2 + 2 * (tptt.p + 1) + 4 * tptt.d_val) * tptt.C.numel()
            for i, tptt in enumerate(self.tptts)
        }
        fpt['Total'] = sum(v for k, v in fpt.items())
        return fpt
