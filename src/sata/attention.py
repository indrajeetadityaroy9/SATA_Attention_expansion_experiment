import math
import torch
from .taylor_term import TightlyPackedTaylorTerm


class SymmetryAwareTaylorApproximatedAttention(torch.nn.Module):
    """
    Self-attention at constant cost per token (Heinsen & Kozachkov, 2026).
    Q [..., n_qry, d_key], K [..., n_tok, d_key], V [..., n_tok, d_val] -> Y [..., n_qry, d_val].
    """

    def __init__(self, d_key, d_val, is_causal, n_taylor=4):
        super().__init__()
        self.d_key, self.d_val, self.is_causal, self.n_taylor = d_key, d_val, is_causal, n_taylor
        self.tptts = torch.nn.ModuleList(TightlyPackedTaylorTerm(d_key, d_val, p, is_causal) for p in range(n_taylor))

    def forward(self, Q, K, V, continue_prev=False):
        S_terms, Z_terms = zip(*(t(Q, K, V, continue_prev) for t in self.tptts))
        return torch.stack(S_terms).sum(0) / torch.stack(Z_terms).sum(0)

    def reset_state(self):
        for t in self.tptts:
            del t.prev_H_S, t.prev_H_Z

    def get_hidden_state_sizes(self):
        szs = {f'tptt[{i}]': math.comb(t.d_key + t.p - 1, t.p) * (t.d_val + 1) for i, t in enumerate(self.tptts)}
        return {**szs, 'Total': sum(szs.values())}

    def get_forward_FLOPs_per_query_head(self):
        fpt = {f'tptt[{i}]': (2 + 2 * (t.p + 1) + 4 * t.d_val) * t.C.numel() for i, t in enumerate(self.tptts)}
        return {**fpt, 'Total': sum(fpt.values())}
