###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# Adapted from Tencent HunyuanVideo-1.5 / HY-WorldPlay official implementation.

import einops
import torch
from typing import Optional
from loguru import logger
import numpy as np
import torch.nn.functional as F

from ..flash_attn_no_pad import (
    flash_attn_no_pad,
    flash_attn_no_pad_v3,
    flex_attn_no_pad
)
from ..runtime_helpers import get_infer_state, maybe_fallback_attn_mode

from ..distributed import (
    get_sp_parallel_rank,
    get_sp_world_size,
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
)

try:
    from torch.nn.attention.flex_attention import flex_attention

    flex_attention = torch.compile(flex_attention, dynamic=False)
    torch._dynamo.config.cache_size_limit = 192
    torch._dynamo.config.accumulated_cache_size_limit = 192
    flex_mask_cache = {}
except Exception:
    logger.warning("Could not load Sliding Tile Attention of FlexAttn.")

from .ssta_attention import ssta_3d_attention

# The chunk-causal mask depends only on sequence geometry, so it is identical for
# every block in a forward pass. Build it once in bool: at 480x832 with 24 latent
# frames the float32 form upstream allocates is 5.4 GiB per call.
_CHUNK_CAUSAL_MASK_CACHE: dict = {}


def _use_aiter_chunk_causal() -> bool:
    try:
        from primus.backends.diffusion.attention import get_attention_backend
        from primus.backends.diffusion.attention.aiter import AITER_FLASH_ATTN_AVAILABLE
    except Exception:
        return False
    return get_attention_backend() == "flash_attn_aiter" and AITER_FLASH_ATTN_AVAILABLE


def _chunk_causal_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    text_seq_length: int,
    chunk_seq_length: int,
) -> torch.Tensor:
    """Chunk-causal AR attention without materializing an S×S mask.

    Text queries attend to text keys. Vision chunk i attends to all text plus
    vision chunks 0..i. Remainder vision tokens attend only to text, matching
    the dense torch_causal mask.
    """
    from primus.backends.diffusion.attention.aiter import aiter_flash_attention

    vision_seq_length = query.shape[1] - text_seq_length
    outputs = [
        aiter_flash_attention(
            query[:, :text_seq_length],
            key[:, :text_seq_length],
            value[:, :text_seq_length],
            dropout_p=0.0,
            causal=False,
            dtype=query.dtype if query.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16,
        )
    ]
    chunk_num = vision_seq_length // chunk_seq_length
    for chunk_index in range(chunk_num):
        start = text_seq_length + chunk_index * chunk_seq_length
        end = start + chunk_seq_length
        outputs.append(
            aiter_flash_attention(
                query[:, start:end],
                key[:, :end],
                value[:, :end],
                dropout_p=0.0,
                causal=False,
                dtype=query.dtype if query.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16,
            )
        )
    remainder_start = text_seq_length + chunk_num * chunk_seq_length
    if remainder_start < query.shape[1]:
        outputs.append(
            aiter_flash_attention(
                query[:, remainder_start:],
                key[:, :text_seq_length],
                value[:, :text_seq_length],
                dropout_p=0.0,
                causal=False,
                dtype=query.dtype if query.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16,
            )
        )
    return torch.cat(outputs, dim=1)


def _chunk_causal_mask(
    total_seq_length: int,
    text_seq_length: int,
    chunk_seq_length: int,
    chunk_num: int,
    device: torch.device,
) -> torch.Tensor:
    key = (total_seq_length, text_seq_length, chunk_seq_length, chunk_num, str(device))
    cached = _CHUNK_CAUSAL_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    causal_mask = torch.zeros(
        (total_seq_length, total_seq_length), device=device, dtype=torch.bool
    )
    causal_mask[:, :text_seq_length] = True  # no attention for the rest
    for i in range(chunk_num):
        start_i = text_seq_length + i * chunk_seq_length
        end_i = min(start_i + chunk_seq_length, total_seq_length)
        for j in range(i + 1):
            start_j = text_seq_length + j * chunk_seq_length
            end_j = min(start_j + chunk_seq_length, total_seq_length)
            # full attention within chunk i for j == i, causal for j < i
            causal_mask[start_i:end_i, start_j:end_j] = True

    _CHUNK_CAUSAL_MASK_CACHE.clear()
    _CHUNK_CAUSAL_MASK_CACHE[key] = causal_mask
    return causal_mask


@torch.compiler.disable
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    drop_rate: float = 0.0,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    attn_mode: str = "flash",
) -> torch.Tensor:
    """
    Compute attention using flash_attn_no_pad or torch scaled_dot_product_attention.

    Args:
        q: Query tensor of shape [B, L, H, D]
        k: Key tensor of shape [B, L, H, D]
        v: Value tensor of shape [B, L, H, D]
        drop_rate: Dropout rate for attention weights.
        attn_mask: Optional attention mask of shape [B, L].
        causal: Whether to apply causal masking.
        attn_mode: Attention mode, either "flash" or "torch". Defaults to "flash".

    Returns:
        Output tensor after attention of shape [B, L, H*D]
    """
    attn_mode = maybe_fallback_attn_mode(attn_mode)

    if attn_mode == "torch":
        # transpose q,k,v dim to fit scaled_dot_product_attention
        query = q.transpose(1, 2)  # B * H * L * D
        key = k.transpose(1, 2)    # B * H * L * D
        value = v.transpose(1, 2)  # B * H * L * D
        
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [torch.int64, torch.int32]:
                assert attn_mask.max() <= 1 and attn_mask.min() >= 0, f'Integer attention mask must be between 0 and 1 for torch attention.'
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError(f'Float attention mask is not implemented for torch attention.')
            attn_mask1 = einops.rearrange(attn_mask, 'b l -> b 1 l 1')
            attn_mask2 = einops.rearrange(attn_mask1, 'b 1 l 1 -> b 1 1 l')
            attn_mask = attn_mask1 & attn_mask2
        
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)
        
        # transpose back
        x = x.transpose(1, 2)  # B * L * H * D
        b, s, h, d = x.shape
        out = x.reshape(b, s, -1)
        return out
    else:
        # flash mode (default)
        qkv = torch.stack([q, k, v], dim=2)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()
        x = flash_attn_no_pad(qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None)
        b, s, a, d = x.shape
        out = x.reshape(b, s, -1)
        return out


@torch.compiler.disable
def parallel_attention(q, k, v, img_q_len, img_kv_len, 
                       attn_mode=None, text_mask=None, 
                       attn_param=None,
                       block_idx=None,
                       ):
    return sequence_parallel_attention(q, k, v, img_q_len, img_kv_len, attn_mode, text_mask, attn_param=attn_param, block_idx=block_idx)


def sequence_parallel_attention(q, k, v, 
                                img_q_len, img_kv_len, 
                                attn_mode=None, text_mask=None,
                                attn_param=None,
                                block_idx=None,
                                ):
    assert attn_mode is not None
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v

    sp_world_size = get_sp_world_size()
    rank_in_sp_group = get_sp_parallel_rank()

    if sp_world_size > 1:
        sp_size = sp_world_size
        sp_rank = rank_in_sp_group
    
        # batch_size, seq_len, attn_heads, head_dim
        query = sequence_model_parallel_all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = sequence_model_parallel_all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = sequence_model_parallel_all_to_all_4D(value, scatter_dim=2, gather_dim=1)

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // sp_size
            return encoder_state.narrow(
                dim, sp_rank * local_heads, local_heads
            )

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    attn_mode = maybe_fallback_attn_mode(attn_mode, get_infer_state(), block_idx)
    
    if attn_mode == "sageattn":
        from sageattention import sageattn
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        hidden_states = sageattn(query, key, value, tensor_layout="NHD", is_causal=False)
    elif attn_mode == "torch":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        if text_mask is not None:
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            attn_mask = None

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [torch.int64, torch.int32]:
                assert attn_mask.max() <= 1 and attn_mask.min() >= 0, f'Integer attention mask must be between 0 and 1 for torch attention.'
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError(f'Float attention mask is not implemented for torch attention.')
            
        # transpose q,k,v dim to fit scaled_dot_product_attention
        query = query.transpose(1, 2)  # B * Head_num * length * dim
        key = key.transpose(1, 2)      # B * Head_num * length * dim
        value = value.transpose(1, 2)  # B * Head_num * length * dim
        if attn_mask is not None:
            attn_mask1 = einops.rearrange(attn_mask, 'b l -> b 1 l 1')
            attn_mask2 = einops.rearrange(attn_mask1, 'b 1 l 1 -> b 1 1 l')
            attn_mask = attn_mask1 & attn_mask2
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        
        # transpose back
        hidden_states = hidden_states.transpose(1, 2)

    # add new attn for chunk-wise attn
    elif attn_mode == "torch_causal":    # now: we set text_mask = None
        # attention: here we concat the encoder text sequence first, then apply causal attention
        vision_seq_length = query.shape[1]
        text_seq_length = encoder_query.shape[1]
        total_seq_length = vision_seq_length + text_seq_length

        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

        # prepare causal mask for chunk-wise attention
        latent_seq_length = 1560      # set for hunyuanvideo 1.5, which is for 480 * 832 resolution
        chunk_seq_length = 1560 * 4
        chunk_num = (vision_seq_length) // chunk_seq_length
        if _use_aiter_chunk_causal():
            hidden_states = _chunk_causal_flash_attention(
                query, key, value, text_seq_length, chunk_seq_length
            )
        else:
            causal_mask = _chunk_causal_mask(
                total_seq_length, text_seq_length, chunk_seq_length, chunk_num, query.device
            )

            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) # 1, 1, S, S
            causal_mask = causal_mask.expand(query.shape[0], 1, -1, -1)

            query = query.transpose(1, 2)  # B * H * L * D
            key = key.transpose(1, 2)      # B * H * L * D
            value = value.transpose(1, 2)  # B * H * L * D

            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=causal_mask, dropout_p=0.0, is_causal=False)

            # transpose back
            hidden_states = hidden_states.transpose(1, 2)   # [B, S, H, D]

        # return back to the original order: [query, encoder_query]
        hidden_states, encoder_hidden_states = hidden_states[:, text_seq_length:, :, :], hidden_states[:, :text_seq_length, :, :]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        
    elif attn_mode == "flash2":
        query = torch.cat([query, encoder_query], dim=1)   # vision token first
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)

        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)  
    
    elif attn_mode == "flex_causal":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)
        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flex_attn_no_pad(qkv, attn_mask, causal=True, dropout_p=0.0, softmax_scale=None)
        
    elif attn_mode == "flash3":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)
        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad_v3(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)

    elif attn_mode == "flex-block-attn":
        sparse_type = attn_param["attn_sparse_type"]  # sta/block_attn/ssta
        ssta_threshold = attn_param["ssta_threshold"]
        ssta_lambda = attn_param["ssta_lambda"]
        ssta_sampling_type = attn_param["ssta_sampling_type"]
        ssta_adaptive_pool = attn_param["ssta_adaptive_pool"]

        attn_pad_type = attn_param["attn_pad_type"]  # repeat/zero
        attn_use_text_mask = attn_param["attn_use_text_mask"]
        attn_mask_share_within_head = attn_param["attn_mask_share_within_head"]

        ssta_topk = attn_param["ssta_topk"]
        thw = attn_param["thw"]
        tile_size = attn_param["tile_size"]
        win_size = attn_param["win_size"][0].copy()

        def get_image_tile(tile_size):
            block_size = np.prod(tile_size)
            if block_size == 384:
                tile_size = (1, 16, 24)
            elif block_size == 128:
                tile_size = (1, 16, 8)
            elif block_size == 64:
                tile_size = (1, 8, 8)
            elif block_size == 16:
                tile_size = (1, 4, 4)
            else:
                raise ValueError(f"Error tile_size {tile_size}, only support in [16, 64, 128, 384]")
            return tile_size

        if thw[0] == 1:
            tile_size = get_image_tile(tile_size)
            win_size = [1, 1, 1]
        elif thw[0] <= 31: # 16fps: 5 * 16 / 4 + 1 = 21; 24fps: 5 * 24 / 4 + 1 = 31
            ssta_topk = ssta_topk // 2

        # Concatenate and permute query, key, value to (B, H, S, D)
        query = torch.cat([query, encoder_query], dim=1).permute(0, 2, 1, 3)
        key = torch.cat([key, encoder_key], dim=1).permute(0, 2, 1, 3)
        value = torch.cat([value, encoder_value], dim=1).permute(0, 2, 1, 3)

        assert (
            query.shape[-1] == 128
        ), "The last dimension of query, key and value must be 128 for flex-block-attn."

        hidden_states = ssta_3d_attention(
            query,
            key,
            value,
            thw,
            topk=ssta_topk,
            tile_thw=tile_size,
            kernel_thw=win_size,
            text_len=encoder_sequence_length,
            sparse_type=sparse_type,
            threshold=ssta_threshold,
            lambda_=ssta_lambda,
            pad_type=attn_pad_type,
            text_mask=text_mask if attn_use_text_mask else None,
            sampling_type=ssta_sampling_type,
            adaptive_pool=ssta_adaptive_pool,
            mask_share_within_head=attn_mask_share_within_head,
        )
        hidden_states, sparse_ratio = hidden_states
        hidden_states = hidden_states.permute(0, 2, 1, 3)

    else:
        raise NotImplementedError(
            f'Unsupported attention mode: {attn_mode}. Only torch, flash, flash3, sageattn and flex-block-attn are supported.'
        )

    if sp_world_size > 1:
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes((sequence_length, encoder_sequence_length), dim=1)
        hidden_states = sequence_model_parallel_all_to_all_4D(hidden_states, scatter_dim=1, gather_dim=2)
        encoder_hidden_states = sequence_model_parallel_all_gather(encoder_hidden_states, dim=2).contiguous()
        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)

    return hidden_states
