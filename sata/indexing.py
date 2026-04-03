import warnings
import math
import torch
from itertools import combinations_with_replacement


def generate_index_matrix(d_key: int, p: int) -> torch.Tensor:
    """
    Generate the index matrix M_p for the upper hyper-triangular region
    of the order-p symmetric tensor. Each row contains indices
    i_1 <= i_2 <= ... <= i_p, selecting the unique monomials.

    Returns tensor of shape [m_p, p] where m_p = C(d_key + p - 1, p).
    """
    return torch.tensor(
        [*combinations_with_replacement(range(d_key), p)],
        dtype=torch.long,
    )


def calculate_multiplicity(M: torch.Tensor, d_key: int) -> torch.Tensor:
    """
    Calculate the number of multiset permutations of each row of M.
    Each entry is p! / (n_1! * n_2! * ... * n_{d_key}!) where n_i is
    the count of index i in the row.

    See https://en.wikipedia.org/wiki/Permutation#Permutations_of_multisets.

    Returns long tensor of shape [m_p].
    """
    log_factorial = lambda counts: torch.lgamma(counts + 1)
    vmap_bincount = torch.vmap(lambda row: torch.bincount(row, minlength=d_key))
    with warnings.catch_warnings(action="ignore"):
        bin_counts = vmap_bincount(M)
    log_numer = log_factorial(torch.tensor(M.size(-1)))
    log_denoms = torch.sum(log_factorial(bin_counts), dim=-1)
    return torch.exp(log_numer - log_denoms).round().long()
