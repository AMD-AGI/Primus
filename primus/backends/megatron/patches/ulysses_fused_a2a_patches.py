###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ulysses context parallelism with Primus-Turbo's fused GEMM + all-to-all ops.

Opt-in with ``PRIMUS_ULYSSES_FUSED_A2A=1`` (CP > 1, TP = 1). Self-attention then runs

    x [s/cp, b, h] --ulysses_qkv_gemm_a2a--> q, k, v [s, b, h/cp, d] (local heads, full sequence)
      -> q/k layernorm -> RoPE over the full sequence -> flash_attn_func (local, causal)
      --ulysses_a2a_o_gemm--> [s/cp, b, h]

so the QKV / o_proj GEMMs overlap their all-to-alls and the pack / unpack copies disappear.
The fused ops shard the sequence contiguously (rank r holds tokens [r*s/cp, (r+1)*s/cp)), so the
CP batch split is switched from Megatron's zigzag to contiguous chunks; with Ulysses every rank
attends over the whole sequence, so the zigzag load balancing buys nothing.
"""

import os

import torch
import torch.nn.functional as F

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_METADATA_KEYS = ("cu_seqlens", "cu_seqlens_padded", "max_seqlen", "local_cp_size", "hybrid_cp_group")


def _enabled(ctx: PatchContext) -> bool:
    if os.environ.get("PRIMUS_ULYSSES_FUSED_A2A", "0") != "1":
        return False
    args = get_args(ctx)
    tp = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    cp = int(getattr(args, "context_parallel_size", 1) or 1)
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    # the fused kernels exchange over XGMI peer memory, so every CP group must sit inside one node;
    # with TP = 1 and Megatron's tp-cp-ep-dp-pp order a CP group is `cp` consecutive ranks
    if tp != 1 or cp == 1 or local_world % cp != 0:
        warning_rank_0(
            f"[Patch:megatron.ulysses_fused_a2a] needs TP=1 and CP>1 dividing {local_world} GPUs/node "
            f"(got TP={tp}, CP={cp}), keeping the stock path"
        )
        return False
    # MLA keeps its own forward and YaRN RoPE slicing, which would disagree with the contiguous split
    if getattr(args, "multi_latent_attention", False):
        warning_rank_0("[Patch:megatron.ulysses_fused_a2a] MLA is not supported, keeping the stock path")
        return False
    return True


def _contiguous_cp_split(batch, cp_group):
    cp_size = torch.distributed.get_world_size(cp_group)
    cp_rank = torch.distributed.get_rank(cp_group)
    if cp_size == 1:
        return batch
    for key, val in batch.items():
        if key in _METADATA_KEYS or val is None:
            continue
        seq_dim = 2 if key == "attention_mask" else 1
        chunk = val.size(seq_dim) // cp_size
        batch[key] = val.narrow(seq_dim, cp_rank * chunk, chunk)
    return batch


class _NoCPGroup:
    """RoPE's cp_group stand-in after the all-to-all: q and k already span the whole sequence."""

    def size(self):
        return 1

    def rank(self):
        return 0


def _a2a_helper(b, seq, hq, hkv, d, n):
    from primus_turbo.pytorch.ops.attention.usp.attention_a2a_helper import (
        get_attention_cp_a2a_helper,
    )

    return get_attention_cp_a2a_helper(b, seq, hq, hkv, d, d, 0, n)


def _a2a(t, group):
    from torch.distributed.nn.functional import all_to_all_single

    return all_to_all_single(torch.empty_like(t), t.contiguous(), group=group)


def _serial_qkv_gemm_a2a(x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout):
    """Unfused ablation of ``ulysses_qkv_gemm_a2a``: GEMM, pack, all-to-all, unpack in series."""
    s_local, b = x.shape[:2]
    helper = _a2a_helper(b, s_local * group.size(), hq, hkv, d, group.size())
    qkv = F.linear(x, w_qkv).view(s_local, b, hkv, hq // hkv + 2, d)
    q = qkv[..., : hq // hkv, :].reshape(s_local, b, hq, d)
    k, v = qkv[..., hq // hkv, :], qkv[..., hq // hkv + 1, :]
    return helper.splits_qkv_after_a2a(_a2a(helper.combine_qkv_before_a2a(q, k, v), group))


def _serial_a2a_o_gemm(o, w_o, residual, group, qkv_format):
    """Unfused ablation of ``ulysses_a2a_o_gemm``."""
    seq, b, heads, d = o.shape
    n = group.size()
    helper = _a2a_helper(b, seq, heads * n, heads * n, d, n)
    o_gathered = helper.reshape_o_after_a2a(_a2a(helper.reshape_o_before_a2a(o), group))
    return F.linear(o_gathered.reshape(*o_gathered.shape[:2], -1), w_o)


def _make_fused_forward(original_forward):
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

    from primus_turbo.pytorch.ops import flash_attn_func
    from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import (
        ulysses_a2a_o_gemm,
        ulysses_qkv_gemm_a2a,
    )

    no_cp = _NoCPGroup()
    if os.environ.get("PRIMUS_ULYSSES_FUSED_A2A_IMPL", "fused") == "serial":
        qkv_op, o_op = _serial_qkv_gemm_a2a, _serial_a2a_o_gemm
    else:
        qkv_op, o_op = ulysses_qkv_gemm_a2a, ulysses_a2a_o_gemm

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        cfg = self.config
        cp_group = self.pg_collection.cp
        if (
            cp_group.size() == 1
            or key_value_states is not None
            or inference_context is not None
            or inference_params is not None
            or packed_seq_params is not None
            or attention_bias is not None
            or cfg.attention_output_gate
            or getattr(self.linear_qkv, "bias", None) is not None
            or getattr(self.linear_proj, "bias", None) is not None
        ):
            return original_forward(
                self,
                hidden_states,
                attention_mask,
                key_value_states=key_value_states,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
            )

        hq, hkv, d = cfg.num_attention_heads, cfg.num_query_groups, cfg.kv_channels
        x = hidden_states
        ln_weight = getattr(self.linear_qkv, "layer_norm_weight", None)
        if ln_weight is not None:
            # the input RMSNorm fused into linear_qkv
            x = F.rms_norm(x, (x.shape[-1],), ln_weight, cfg.layernorm_epsilon)
        q, k, v = qkv_op(x, self.linear_qkv.weight, hq, hkv, d, cp_group, "sbhd", "interleaved")
        if self.q_layernorm is not None:
            q = self.q_layernorm(q)
        if self.k_layernorm is not None:
            k = self.k_layernorm(k)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = (
                rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else (rotary_pos_emb,) * 2
            )
            q = apply_rotary_pos_emb(q, q_pos_emb, config=cfg, cp_group=no_cp)
            k = apply_rotary_pos_emb(k, k_pos_emb, config=cfg, cp_group=no_cp)
        # sbhd -> bshd views for flash-attn
        o = flash_attn_func(
            q.permute(1, 0, 2, 3),
            k.permute(1, 0, 2, 3),
            v.permute(1, 0, 2, 3),
            dropout_p=0.0,
            softmax_scale=None,
            causal=True,
        )
        o = o.permute(1, 0, 2, 3).contiguous()
        y = o_op(o, self.linear_proj.weight, None, cp_group, "sbhd")
        return y, None

    return forward


@register_patch(
    "megatron.ulysses_fused_a2a",
    backend="megatron",
    phase="before_train",
    description="Ulysses CP through Primus-Turbo GemmA2A / A2AGemm (PRIMUS_ULYSSES_FUSED_A2A=1)",
    condition=_enabled,
)
def patch_ulysses_fused_a2a(ctx: PatchContext):
    import megatron.core.models.common.embeddings.rotary_pos_embedding as rope_mod
    import megatron.core.utils as mcore_utils
    from megatron.core.transformer.attention import SelfAttention

    mcore_utils._get_batch_on_this_cp_rank_per_sequence_balancing = _contiguous_cp_split
    # RoPE runs after the all-to-all over the full sequence, so keep the full-length freqs
    rope_mod.get_pos_emb_on_this_cp_rank = lambda emb, seq_dim, cp_group: emb
    SelfAttention.forward = _make_fused_forward(SelfAttention.forward)
    log_rank_0("[Patch:megatron.ulysses_fused_a2a] SelfAttention -> GemmA2A / A2AGemm, contiguous CP split")
