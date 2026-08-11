# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from wan.modules.attention import attention
from wan.modules.flash_attention_backend import FlashAttentionBackend
from wan.modules.hsa import HSAAttention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import time
import torch.distributed as dist
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller


# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 block_idx=-1):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.block_idx = block_idx
        # Support list/tuple local_attn_size by converting to list first (handles OmegaConf ListConfig)
        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, "__iter__"):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.hsa_attention = HSAAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            block_size=64,
            tau_min=0.35,
            tau_max=1.0,
            proj_segment_len=1560,
        )
        self.flash_attention = None
        self.kv_cache_attn_backend = "hsa"
        self._kv_cache_backend_logged = False
        self.use_hsa_kv_cache = False

    def set_kv_cache_attn_backend(
        self,
        backend: str,
        *,
        hsa_backend: str = "auto",
    ) -> None:
        """Choose the KV-cache self-attention: HSA, or dense FlashAttention as the baseline."""
        backend = str(backend).lower().strip()
        if backend not in {"hsa", "flash"}:
            raise ValueError(f"Unsupported KV-cache attention backend '{backend}'. Expected one of: hsa, flash.")

        self.hsa_attention.backend = hsa_backend
        if backend == "flash" and self.flash_attention is None:
            flash_attention = FlashAttentionBackend(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            )
            self.flash_attention = flash_attention.to(device=self.q.weight.device, dtype=self.q.weight.dtype)

        self.kv_cache_attn_backend = backend
        self._kv_cache_backend_logged = False

    def get_kv_cache_attn_module(self):
        if self.kv_cache_attn_backend == "flash":
            return self.flash_attention
        return self.hsa_attention

    def uses_hsa_coarse_cache(self) -> bool:
        return self.use_hsa_kv_cache and self.kv_cache_attn_backend == "hsa"

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        layer_index=None,
        sink_recache_after_switch=False
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start
        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                def run_rope_tf():
                    q_chunk = torch.chunk(q, 2, dim=1)
                    k_chunk = torch.chunk(k, 2, dim=1)
                    roped_query_local = []
                    roped_key_local = []
                    for ii in range(2):
                        rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                        rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                        roped_query_local.append(rq)
                        roped_key_local.append(rk)
                    return torch.cat(roped_query_local, dim=1), torch.cat(roped_key_local, dim=1)

                roped_query, roped_key = run_rope_tf()

                def run_cache_prepare_tf():
                    padded_length_local = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                    padded_roped_query_local = torch.cat(
                        [roped_query,
                         torch.zeros([q.shape[0], padded_length_local, q.shape[2], q.shape[3]],
                                     device=q.device, dtype=v.dtype)],
                        dim=1
                    )

                    padded_roped_key_local = torch.cat(
                        [roped_key, torch.zeros([k.shape[0], padded_length_local, k.shape[2], k.shape[3]],
                                                device=k.device, dtype=v.dtype)],
                        dim=1
                    )

                    padded_v_local = torch.cat(
                        [v, torch.zeros([v.shape[0], padded_length_local, v.shape[2], v.shape[3]],
                                        device=v.device, dtype=v.dtype)],
                        dim=1
                    )
                    return padded_length_local, padded_roped_query_local, padded_roped_key_local, padded_v_local

                padded_length, padded_roped_query, padded_roped_key, padded_v = run_cache_prepare_tf()

                def run_flex_attn_tf():
                    return flex_attention(
                        query=padded_roped_query.transpose(2, 1),
                        key=padded_roped_key.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask
                    )[:, :, :-padded_length].transpose(2, 1)

                x = run_flex_attn_tf()

            else:
                def run_rope():
                    roped_query_local = rope_apply(q, grid_sizes, freqs).type_as(v)
                    roped_key_local = rope_apply(k, grid_sizes, freqs).type_as(v)
                    return roped_query_local, roped_key_local

                roped_query, roped_key = run_rope()

                def run_cache_prepare():
                    padded_length_local = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                    padded_roped_query_local = torch.cat(
                        [roped_query,
                         torch.zeros([q.shape[0], padded_length_local, q.shape[2], q.shape[3]],
                                     device=q.device, dtype=v.dtype)],
                        dim=1
                    )

                    padded_roped_key_local = torch.cat(
                        [roped_key, torch.zeros([k.shape[0], padded_length_local, k.shape[2], k.shape[3]],
                                                device=k.device, dtype=v.dtype)],
                        dim=1
                    )

                    padded_v_local = torch.cat(
                        [v, torch.zeros([v.shape[0], padded_length_local, v.shape[2], v.shape[3]],
                                        device=v.device, dtype=v.dtype)],
                        dim=1
                    )
                    return padded_length_local, padded_roped_query_local, padded_roped_key_local, padded_v_local

                padded_length, padded_roped_query, padded_roped_key, padded_v = run_cache_prepare()

                def run_flex_attn():
                    return flex_attention(
                        query=padded_roped_query.transpose(2, 1),
                        key=padded_roped_key.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask
                    )[:, :, :-padded_length].transpose(2, 1)

                x = run_flex_attn()
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            def run_causal_rope():
                roped_query_local = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                roped_key_local = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                return roped_query_local, roped_key_local

            roped_query, roped_key = run_causal_rope()

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]

            def run_cache_prepare_kv():
                cache_update_info_local = None
                use_hsa_coarse_cache_local = (
                    self.uses_hsa_coarse_cache()
                    and "k_coarse" in kv_cache
                    and "v_coarse" in kv_cache
                )

                def clone_cache_tensors():
                    temp_k_local = kv_cache["k"].clone()
                    temp_v_local = kv_cache["v"].clone()
                    temp_k_coarse_local = kv_cache["k_coarse"].clone() if use_hsa_coarse_cache_local else None
                    temp_v_coarse_local = kv_cache["v_coarse"].clone() if use_hsa_coarse_cache_local else None
                    return temp_k_local, temp_v_local, temp_k_coarse_local, temp_v_coarse_local

                def write_coarse_cache(
                    temp_k_coarse_local,
                    temp_v_coarse_local,
                    write_start_index_local,
                    write_end_index_local,
                    new_k_local,
                    new_v_local,
                ):
                    if (
                        not use_hsa_coarse_cache_local
                        or temp_k_coarse_local is None
                        or temp_v_coarse_local is None
                        or write_end_index_local <= write_start_index_local
                    ):
                        return None, None

                    coarse_write_start, coarse_write_end = self.hsa_attention.token_range_to_coarse(
                        int(write_start_index_local),
                        int(write_end_index_local),
                    )
                    new_k_coarse_local, new_v_coarse_local = self.hsa_attention.compress_kv_cache(
                        new_k_local,
                        new_v_local,
                    )
                    temp_k_coarse_local[:, coarse_write_start:coarse_write_end] = new_k_coarse_local
                    temp_v_coarse_local[:, coarse_write_start:coarse_write_end] = new_v_coarse_local
                    return new_k_coarse_local, new_v_coarse_local

                is_recompute_local = current_end <= kv_cache["global_end_index"].item() and current_start > 0
                if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                        num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                    num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                    num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens

                    local_end_index_local = kv_cache["local_end_index"].item() + current_end - \
                        kv_cache["global_end_index"].item() - num_evicted_tokens
                    local_start_index_local = local_end_index_local - num_new_tokens

                    temp_k_local, temp_v_local, temp_k_coarse_local, temp_v_coarse_local = clone_cache_tensors()

                    temp_k_local[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        temp_k_local[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    temp_v_local[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        temp_v_local[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    if use_hsa_coarse_cache_local and temp_k_coarse_local is not None and temp_v_coarse_local is not None:
                        _, coarse_sink_tokens = self.hsa_attention.token_range_to_coarse(0, int(sink_tokens))
                        coarse_roll_src_start, coarse_roll_src_end = self.hsa_attention.token_range_to_coarse(
                            int(sink_tokens + num_evicted_tokens),
                            int(sink_tokens + num_evicted_tokens + num_rolled_tokens),
                        )
                        coarse_num_rolled = coarse_roll_src_end - coarse_roll_src_start
                        if coarse_num_rolled > 0:
                            temp_k_coarse_local[:, coarse_sink_tokens:coarse_sink_tokens + coarse_num_rolled] = \
                                temp_k_coarse_local[:, coarse_roll_src_start:coarse_roll_src_end].clone()
                            temp_v_coarse_local[:, coarse_sink_tokens:coarse_sink_tokens + coarse_num_rolled] = \
                                temp_v_coarse_local[:, coarse_roll_src_start:coarse_roll_src_end].clone()

                    write_start_index_local = max(local_start_index_local, sink_tokens) if is_recompute_local else local_start_index_local
                    roped_offset_local = max(0, write_start_index_local - local_start_index_local)
                    write_len_local = max(0, local_end_index_local - write_start_index_local)
                    new_k_local = roped_key[:, roped_offset_local:roped_offset_local + write_len_local]
                    new_v_local = v[:, roped_offset_local:roped_offset_local + write_len_local]
                    if write_len_local > 0:
                        temp_k_local[:, write_start_index_local:local_end_index_local] = new_k_local
                        temp_v_local[:, write_start_index_local:local_end_index_local] = new_v_local
                    new_k_coarse_local, new_v_coarse_local = write_coarse_cache(
                        temp_k_coarse_local,
                        temp_v_coarse_local,
                        write_start_index_local,
                        local_end_index_local,
                        new_k_local,
                        new_v_local,
                    )

                    cache_update_info_local = {
                        "action": "roll_and_insert",
                        "sink_tokens": sink_tokens,
                        "num_rolled_tokens": num_rolled_tokens,
                        "num_evicted_tokens": num_evicted_tokens,
                        "local_start_index": local_start_index_local,
                        "local_end_index": local_end_index_local,
                        "write_start_index": write_start_index_local,
                        "write_end_index": local_end_index_local,
                        "new_k": new_k_local,
                        "new_v": new_v_local,
                        "new_k_coarse": new_k_coarse_local,
                        "new_v_coarse": new_v_coarse_local,
                        "current_end": current_end,
                        "is_recompute": is_recompute_local
                    }
                else:
                    local_end_index_local = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                    local_start_index_local = local_end_index_local - num_new_tokens

                    temp_k_local, temp_v_local, temp_k_coarse_local, temp_v_coarse_local = clone_cache_tensors()
                    write_start_index_local = max(local_start_index_local, sink_tokens) if is_recompute_local else local_start_index_local
                    if sink_recache_after_switch:
                        write_start_index_local = local_start_index_local
                    roped_offset_local = max(0, write_start_index_local - local_start_index_local)
                    write_len_local = max(0, local_end_index_local - write_start_index_local)
                    new_k_local = roped_key[:, roped_offset_local:roped_offset_local + write_len_local]
                    new_v_local = v[:, roped_offset_local:roped_offset_local + write_len_local]
                    if write_len_local > 0:
                        temp_k_local[:, write_start_index_local:local_end_index_local] = new_k_local
                        temp_v_local[:, write_start_index_local:local_end_index_local] = new_v_local
                    new_k_coarse_local, new_v_coarse_local = write_coarse_cache(
                        temp_k_coarse_local,
                        temp_v_coarse_local,
                        write_start_index_local,
                        local_end_index_local,
                        new_k_local,
                        new_v_local,
                    )

                    cache_update_info_local = {
                        "action": "direct_insert",
                        "local_start_index": local_start_index_local,
                        "local_end_index": local_end_index_local,
                        "write_start_index": write_start_index_local,
                        "write_end_index": local_end_index_local,
                        "new_k": new_k_local,
                        "new_v": new_v_local,
                        "new_k_coarse": new_k_coarse_local,
                        "new_v_coarse": new_v_coarse_local,
                        "current_end": current_end,
                        "is_recompute": is_recompute_local
                    }
                return (
                    temp_k_local,
                    temp_v_local,
                    temp_k_coarse_local,
                    temp_v_coarse_local,
                    cache_update_info_local,
                    local_end_index_local,
                )

            temp_k, temp_v, temp_k_coarse, temp_v_coarse, cache_update_info, local_end_index = run_cache_prepare_kv()


            def _kv_cache_attention(
                q_tokens: torch.Tensor,
                k_tokens: torch.Tensor,
                v_tokens: torch.Tensor,
                k_coarse_tokens: torch.Tensor | None = None,
                v_coarse_tokens: torch.Tensor | None = None,
            ) -> torch.Tensor:
                if not self.use_hsa_kv_cache:
                    return attention(q_tokens, k_tokens, v_tokens)

                backend_name = self.kv_cache_attn_backend.upper()
                backend_module = self.get_kv_cache_attn_module()
                if backend_module is None:
                    raise RuntimeError(f"KV-cache backend '{self.kv_cache_attn_backend}' is not initialized.")
                if (not self._kv_cache_backend_logged) and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[DiT KV] Using {backend_name} backend for KV-cache self-attention.")
                    self._kv_cache_backend_logged = True


                if self.kv_cache_attn_backend == "hsa":
                    return backend_module(
                        q_tokens,
                        k_tokens,
                        v_tokens,
                        k_coarse=k_coarse_tokens,
                        v_coarse=v_coarse_tokens,
                    )

                return backend_module(
                    q_tokens,
                    k_tokens,
                    v_tokens,
                )

            def _slice_coarse_cache(start_index: int, end_index: int):
                if temp_k_coarse is None or temp_v_coarse is None:
                    return None, None
                coarse_start, coarse_end = self.hsa_attention.token_range_to_coarse(
                    int(start_index),
                    int(end_index),
                )
                return (
                    temp_k_coarse[:, coarse_start:coarse_end],
                    temp_v_coarse[:, coarse_start:coarse_end],
                )

            def run_hsa_core_attn():
                if sink_tokens > 0:
                    local_budget = self.max_attention_size - sink_tokens
                    k_sink = temp_k[:, :sink_tokens]
                    v_sink = temp_v[:, :sink_tokens]
                    k_coarse_sink, v_coarse_sink = _slice_coarse_cache(0, sink_tokens)
                    if local_budget > 0:
                        local_start_for_window = max(sink_tokens, local_end_index - local_budget)
                        k_local = temp_k[:, local_start_for_window:local_end_index]
                        v_local = temp_v[:, local_start_for_window:local_end_index]
                        k_cat = torch.cat([k_sink, k_local], dim=1)
                        v_cat = torch.cat([v_sink, v_local], dim=1)
                        k_coarse_local, v_coarse_local = _slice_coarse_cache(local_start_for_window, local_end_index)
                        if k_coarse_sink is not None and v_coarse_sink is not None:
                            k_coarse_cat = torch.cat([k_coarse_sink, k_coarse_local], dim=1)
                            v_coarse_cat = torch.cat([v_coarse_sink, v_coarse_local], dim=1)
                        else:
                            k_coarse_cat, v_coarse_cat = None, None
                    else:
                        k_cat = k_sink
                        v_cat = v_sink
                        k_coarse_cat, v_coarse_cat = k_coarse_sink, v_coarse_sink
                    return _kv_cache_attention(
                        roped_query,
                        k_cat,
                        v_cat,
                        k_coarse_tokens=k_coarse_cat,
                        v_coarse_tokens=v_coarse_cat,
                    )

                window_start = max(0, local_end_index - self.max_attention_size)
                attn_k = temp_k[:, window_start:local_end_index]
                attn_v = temp_v[:, window_start:local_end_index]
                attn_k_coarse, attn_v_coarse = _slice_coarse_cache(window_start, local_end_index)
                return _kv_cache_attention(
                    roped_query,
                    attn_k,
                    attn_v,
                    k_coarse_tokens=attn_k_coarse,
                    v_coarse_tokens=attn_v_coarse,
                )

            x = run_hsa_core_attn()

        # output
        def run_out_proj():
            x_local = x.flatten(2)
            return self.o(x_local)

        x = run_out_proj()
        
        # Return both output and cache update info
        if kv_cache is not None:
            return x, (current_end, local_end_index, cache_update_info)
        else:
            return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_idx=-1):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.block_idx = block_idx

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps, block_idx=block_idx)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        layer_index=None,
        sink_recache_after_switch=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        def run_self_attn():
            return self.self_attn(
                (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
                seq_lens, grid_sizes,
                freqs, block_mask, kv_cache, current_start, cache_start, layer_index, sink_recache_after_switch)

        self_attn_result = run_self_attn()
        
        if kv_cache is not None:
            y, cache_update_info = self_attn_result
        else:
            y = self_attn_result
            cache_update_info = None

        # with amp.autocast(dtype=torch.float32):
        def run_residual_misc():
            return x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        x = run_residual_misc()

        def run_cross_attn():
            return x + self.cross_attn(self.norm3(x), context,
                                       context_lens, crossattn_cache=crossattn_cache)

        x = run_cross_attn()

        def run_ffn():
            y_local = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            return x + (y_local.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)

        x = run_ffn()
        
        if cache_update_info is not None:
            # cache_update_info is already in the format (current_end, local_end_index, cache_update_info)
            return x, cache_update_info
        else:
            return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps,
                                    block_idx=block_idx)
            for block_idx in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        #     num_frames = 9
        #     frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)


        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            pass

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _apply_cache_updates(self, kv_cache, cache_update_infos):
        """
        Applies cache updates collected from multiple blocks.
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """
        def update_coarse_cache(cache, update_info):
            if "k_coarse" not in cache or "v_coarse" not in cache:
                return

            segment_len = int(cache.get("coarse_segment_len", 0))
            proj_rank = int(cache.get("coarse_proj_rank", 0))
            if segment_len <= 0 or proj_rank <= 0:
                return

            def to_coarse_index(token_index: int) -> int | None:
                if token_index % segment_len != 0:
                    return None
                return (token_index // segment_len) * proj_rank

            action = update_info["action"]
            new_k_coarse = update_info.get("new_k_coarse")
            new_v_coarse = update_info.get("new_v_coarse")

            if action == "roll_and_insert":
                sink_tokens = int(update_info["sink_tokens"])
                num_evicted_tokens = int(update_info["num_evicted_tokens"])
                num_rolled_tokens = int(update_info["num_rolled_tokens"])

                coarse_sink_tokens = to_coarse_index(sink_tokens)
                coarse_roll_src_start = to_coarse_index(sink_tokens + num_evicted_tokens)
                coarse_roll_src_end = to_coarse_index(sink_tokens + num_evicted_tokens + num_rolled_tokens)
                if None not in {coarse_sink_tokens, coarse_roll_src_start, coarse_roll_src_end}:
                    coarse_num_rolled = int(coarse_roll_src_end - coarse_roll_src_start)
                    if coarse_num_rolled > 0:
                        cache["k_coarse"][:, coarse_sink_tokens:coarse_sink_tokens + coarse_num_rolled] = \
                            cache["k_coarse"][:, coarse_roll_src_start:coarse_roll_src_end].clone()
                        cache["v_coarse"][:, coarse_sink_tokens:coarse_sink_tokens + coarse_num_rolled] = \
                            cache["v_coarse"][:, coarse_roll_src_start:coarse_roll_src_end].clone()

            write_start_index = int(update_info.get("write_start_index", update_info["local_start_index"]))
            write_end_index = int(update_info.get("write_end_index", update_info["local_end_index"]))
            coarse_write_start = to_coarse_index(write_start_index)
            coarse_write_end = to_coarse_index(write_end_index)
            if (
                new_k_coarse is not None
                and new_v_coarse is not None
                and coarse_write_start is not None
                and coarse_write_end is not None
                and coarse_write_end > coarse_write_start
                and new_k_coarse.shape[1] == (coarse_write_end - coarse_write_start)
            ):
                cache["k_coarse"][:, coarse_write_start:coarse_write_end] = new_k_coarse
                cache["v_coarse"][:, coarse_write_start:coarse_write_end] = new_v_coarse

        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            if update_info is not None:
                cache = kv_cache[block_index]
                
                if update_info["action"] == "roll_and_insert":
                    # Apply rolling update
                    sink_tokens = update_info["sink_tokens"]
                    num_rolled_tokens = update_info["num_rolled_tokens"]
                    num_evicted_tokens = update_info["num_evicted_tokens"]
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    
                    # Perform the rolling operation
                    cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    
                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
                    update_coarse_cache(cache, update_info)
                    
                elif update_info["action"] == "direct_insert":
                    # Direct insert
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    
                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
                    update_coarse_cache(cache, update_info)
            
            # Update indices: do not roll back pointers during recomputation
            is_recompute = False if update_info is None else update_info.get("is_recompute", False)
            if not is_recompute:
                kv_cache[block_index]["global_end_index"].fill_(current_end)
                kv_cache[block_index]["local_end_index"].fill_(local_end_index)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        sink_recache_after_switch=False
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        

        def run_patch_embedding():
            return [self.patch_embedding(u.unsqueeze(0)) for u in x]

        x = run_patch_embedding()
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        def run_time_embedding():
            e_local = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
            e0_local = self.time_projection(e_local).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            return e_local, e0_local

        e, e0 = run_time_embedding()
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32
        # context
        context_lens = None
        def run_text_embedding():
            return self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))

        context = run_text_embedding()
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            sink_recache_after_switch=sink_recache_after_switch
        )
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        cache_update_info = None
        cache_update_infos = []  # Collect cache update info for all blocks
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                result = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Extract base info for subsequent blocks (without concrete cache update details)
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                result = block(x, **kwargs)
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Extract base info for subsequent blocks (without concrete cache update details)
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result
        # After all blocks are processed, apply cache updates in a single pass
        if kv_cache is not None and cache_update_infos:
            self._apply_cache_updates(kv_cache, cache_update_infos)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        pass
        raise NotImplementedError()
    
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)
        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # HSAAttention owns how its own parameters start out; re-apply that after the generic
        # Linear initialization above, which would otherwise reach the fusion gate.
        for m in self.modules():
            if isinstance(m, HSAAttention):
                m.reset_parameters()

        # init output layer
        nn.init.zeros_(self.head.head.weight)
