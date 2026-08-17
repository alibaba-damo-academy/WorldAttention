"""Block-sparse attention kernel of the HSA sparse branch. Not included in this release."""
import torch


def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    block_sizes: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Attention of ``[B, H, N, D]`` queries over the key blocks each query block selected."""
    raise NotImplementedError("the HSA block-sparse kernel is not included in this release")
