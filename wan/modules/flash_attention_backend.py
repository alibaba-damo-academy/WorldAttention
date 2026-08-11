import torch
import torch.nn as nn

from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE, flash_attention


class FlashAttentionBackend(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        compute_dtype: torch.dtype = torch.bfloat16,
        fa_version: int | None = None,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.compute_dtype = compute_dtype
        self.fa_version = fa_version
        self.kernel = "flash"

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return flash_kv_attention(
            q,
            k,
            v,
            compute_dtype=self.compute_dtype,
            fa_version=self.fa_version,
        )


def flash_kv_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
    fa_version: int | None = None,
) -> torch.Tensor:
    if q.device.type != "cuda":
        raise RuntimeError("Flash backend requires CUDA tensors.")
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        raise RuntimeError("Flash backend requires flash_attn to be installed.")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Flash backend expects q/k/v with shape [B, L, H, D].")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have the same shape, got {tuple(k.shape)} and {tuple(v.shape)}")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError(
            "Flash backend expects q/k/v to share batch, num_heads, and head_dim, got "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )

    output = flash_attention(
        q=q,
        k=k,
        v=v,
        causal=False,
        dtype=compute_dtype,
        version=fa_version,
    )
    return output.to(q.dtype)
