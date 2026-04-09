import math
import torch
from itertools import combinations_with_replacement


def generate_index_matrix(d_key, p):
    """Index matrix M_p [C(d+p-1,p), p] enumerating unique monomials i_1 <= ... <= i_p."""
    return torch.tensor([*combinations_with_replacement(range(d_key), p)], dtype=torch.long)


def calculate_multiplicity(M, d_key):
    """Multiset permutation counts p!/(n_1!*...*n_d!) per row of M. Returns long [m_p]."""
    log_fac = lambda c: torch.lgamma(c.double() + 1)
    bin_counts = torch.zeros(M.size(0), d_key, dtype=torch.long).scatter_add_(1, M, torch.ones_like(M))
    return torch.exp(log_fac(torch.tensor(M.size(-1), dtype=torch.float64)) - log_fac(bin_counts).sum(-1)).round().long()


class TightlyPackedTaylorTerm(torch.nn.Module):
    """One term (order p) of the symmetry-aware Taylor expansion (Heinsen & Kozachkov, 2026)."""

    def __init__(self, d_key, d_val, p, is_causal):
        super().__init__()
        self.d_key, self.d_val, self.p, self.is_causal = d_key, d_val, p, is_causal
        self.register_buffer('alpha', torch.tensor(1.0 / (math.factorial(p) * d_key ** (p / 2))))
        self.register_buffer('M', generate_index_matrix(d_key, p))
        self.register_buffer('C', calculate_multiplicity(self.M, d_key).double())
        self._accumulate = torch.cumsum if is_causal else lambda x, **_: x.sum(dim=-3, keepdim=True)

    def forward(self, Q, K, V, continue_prev):
        acc = lambda x: self._accumulate(x, dim=-3)
        if self.p == 0:
            H_S, H_Z = acc(V[..., None, :]) * self.alpha, acc(torch.ones_like(V[..., None, :1])) * self.alpha
        else:
            Phi_Q, Phi_K = Q[..., self.M].prod(-1), K[..., self.M].prod(-1)
            H_S, H_Z = acc(Phi_K[..., None] * V[..., None, :]) * self.alpha, acc(Phi_K[..., None]) * self.alpha

        if continue_prev:
            H_S, H_Z = self.prev_H_S + H_S, self.prev_H_Z + H_Z
        self.prev_H_S, self.prev_H_Z = H_S[..., -1:, :, :].detach(), H_Z[..., -1:, :, :].detach()

        if self.p == 0:
            return H_S.squeeze(-2).expand(*Q.shape[:-1], -1), H_Z.squeeze(-2).expand(*Q.shape[:-1], -1)
        C = self.C.to(Phi_Q.dtype)
        return torch.einsum('m,...m,...md->...d', C, Phi_Q, H_S), torch.einsum('m,...m,...md->...d', C, Phi_Q, H_Z)
