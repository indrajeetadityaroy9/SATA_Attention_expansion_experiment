import torch


def phi(x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    Feature map Phi_p(x) that tightly packs the unique monomials of the
    upper hyper-triangular region into a vector.

    x[..., M] gathers elements indexed by M, then prod(dim=-1) multiplies
    across the p indices in each row of M.

    Input x: [..., d_key], output: [..., m_p].

    Note: x[..., M] returns copies, not views (PyTorch advanced indexing).
    """
    return x[..., M].prod(dim=-1)
