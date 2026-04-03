import math
import torch

from .indexing import generate_index_matrix, calculate_multiplicity
from .feature_map import phi


class TightlyPackedTaylorTerm(torch.nn.Module):
    """
    Tightly packed Taylor numerator and denominator terms, as proposed in
    "Self-Attention at Constant Cost per Token via Symmetry-Aware Taylor
    Approximation" (Heinsen and Kozachkov, 2026).

    Args:
        d_key: number of elements per head in queries and keys.
        d_val: number of elements per head in values.
        p: power of Taylor term (order of symmetric tensor products).
        is_causal: if True, computes autoregressive attention.

    Inputs:
        Q: float tensor of shape [..., n_qry, d_key].
        K: float tensor of shape [..., n_tok, d_key].
        V: float tensor of shape [..., n_tok, d_val].
        continue_prev: bool, if True, continues the sequence.

    Outputs:
        S_term: float tensor of shape [..., n_qry, d_val].
        Z_term: float tensor of shape [..., n_qry, 1].
    """

    def __init__(self, d_key: int, d_val: int, p: int, is_causal: bool) -> None:
        super().__init__()
        self.d_key, self.d_val, self.p, self.is_causal = (d_key, d_val, p, is_causal)
        self.register_buffer('alpha', torch.tensor(1.0 / (math.factorial(p) * (d_key ** (p / 2)))))
        self.register_buffer('M', generate_index_matrix(d_key, p))
        self.register_buffer('C', calculate_multiplicity(self.M, d_key).float())
        assert len(self.M) == math.comb(d_key + p - 1, p) and self.C.long().sum() == d_key ** p

    def accumulate(self, summands: torch.Tensor) -> torch.Tensor:
        if self.is_causal:
            return torch.cumsum(summands, dim=-3)
        else:
            return torch.sum(summands, dim=-3, keepdim=True)

    def Phi(self, x: torch.Tensor) -> torch.Tensor:
        return phi(x, self.M)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        continue_prev: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert Q.size(-1) == K.size(-1) and K.size(-2) == V.size(-2), (
            "Input shapes are incompatible. See docstring for required input shapes.")
        if self.is_causal:
            assert Q.size(-2) == K.size(-2) == V.size(-2), (
                "Number of queries, keys, and values must match for causal attention.")

        if self.p == 0:
            H_S_summands = V[..., None, :]                                    # [..., n_tok, 1, d_val]
            H_Z_summands = torch.ones_like(V[..., None, :1])                  # [..., n_tok, 1, 1]

            H_S = self.accumulate(H_S_summands)                               # [..., (n or 1), 1, d_val]
            H_Z = self.accumulate(H_Z_summands)                               # [..., (n or 1), 1, 1]

            if continue_prev:
                H_S = self.prev_H_S + H_S
                H_Z = self.prev_H_Z + H_Z

            S_term = H_S.squeeze(-2).expand(*Q.shape[:-1], -1)               # [..., n_qry, d_val]
            Z_term = H_Z.squeeze(-2).expand(*Q.shape[:-1], -1)               # [..., n_qry, 1]
        else:
            Phi_Q = self.Phi(Q)                                               # [..., n_qry, m]
            Phi_K = self.Phi(K)                                               # [..., n_tok, m]

            H_S_summands = Phi_K[..., None] * V[..., None, :]                # [..., n_tok, m, d_val]
            H_Z_summands = Phi_K[..., None]                                   # [..., n_tok, m, 1]

            H_S = self.accumulate(H_S_summands) * self.alpha                  # [..., (n or 1), m, d_val]
            H_Z = self.accumulate(H_Z_summands) * self.alpha                  # [..., (n or 1), m, 1]

            if continue_prev:
                H_S = self.prev_H_S + H_S
                H_Z = self.prev_H_Z + H_Z

            S_term = torch.einsum('m,...m,...md->...d', self.C, Phi_Q, H_S)   # [..., n_qry, d_val]
            Z_term = torch.einsum('m,...m,...md->...d', self.C, Phi_Q, H_Z)   # [..., n_qry, 1]

        self.prev_H_S = H_S[..., -1:, :, :].detach()                         # [..., 1, m, d_val]
        self.prev_H_Z = H_Z[..., -1:, :, :].detach()                         # [..., 1, m, 1]

        return S_term, Z_term
