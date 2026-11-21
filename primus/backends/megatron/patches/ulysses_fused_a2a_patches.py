###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ulysses context parallelism with Primus-Turbo's fused GEMM + all-to-all ops.

Opt-in with ``turbo_ulysses_fused_a2a: 1`` (CP > 1, TP = 1; ``-1`` runs the unfused serial
ablation on the same path). Self-attention then runs

    x [s/cp, b, h] --ulysses_qkv_gemm_a2a--> q, k, v [s, b, h/cp, d] (local heads, full sequence)
      -> q/k layernorm -> RoPE over the full sequence -> flash_attn_func (local, causal)
      --ulysses_a2a_o_gemm--> [s/cp, b, h]

so the QKV / o_proj GEMMs overlap their all-to-alls and the pack / unpack copies disappear.
The fused ops shard the sequence contiguously (rank r holds tokens [r*s/cp, (r+1)*s/cp)), so the
CP batch split is switched from Megatron's zigzag to contiguous chunks; with Ulysses every rank
attends over the whole sequence, so the zigzag load balancing buys nothing.

The split and the RoPE slicing are global, so no layer can fall back to the stock attention once
the patch is in: every unsupported setting known from the arguments keeps the whole run on the stock
path, and the ones only seen at call time (packed sequences, a non-causal mask, ...) raise.
"""

import os

import torch
import torch.nn.functional as F

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_METADATA_KEYS = ("cu_seqlens", "cu_seqlens_padded", "max_seqlen", "local_cp_size", "hybrid_cp_group")


def _mode(args) -> int:
    mode = int(getattr(args, "turbo_ulysses_fused_a2a", 0) or 0)
    valid = (-1, 0, 1)
    assert mode in valid, f"turbo_ulysses_fused_a2a must be -1 (serial), 0 (off) or 1 (fused), got {mode}"
    return mode


_UNSUPPORTED_ARGS = (
    # MLA keeps its own forward and YaRN RoPE slicing, which would disagree with the contiguous split
    ("multi_latent_attention", "MLA"),
    ("experimental_attention_variant", "experimental attention variants"),
    ("add_qkv_bias", "a QKV bias"),
    ("add_bias_linear", "linear biases"),
    ("attention_output_gate", "the attention output gate"),
    ("window_size", "sliding-window attention"),
    ("attention_dropout", "attention dropout"),
)


def _unsupported(args):
    """Why this run keeps the stock CP path, or None."""
    tp = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    cp = int(getattr(args, "context_parallel_size", 1) or 1)
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    # the fused kernels exchange over XGMI peer memory, so every CP group must sit inside one node;
    # with TP = 1 and Megatron's tp-cp-ep-dp-pp order a CP group is `cp` consecutive ranks
    if tp != 1 or cp == 1 or local_world % cp != 0:
        return f"needs TP=1 and CP>1 dividing {local_world} GPUs/node (got TP={tp}, CP={cp})"
    for name, what in _UNSUPPORTED_ARGS:
        if getattr(args, name, None):
            return f"{what} ({name}={getattr(args, name)}) is not supported"
    normalization = getattr(args, "normalization", "LayerNorm")
    if normalization not in ("RMSNorm", "LayerNorm"):
        return f"normalization={normalization} is not supported"
    return None


def _enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    if _mode(args) == 0:
        return False
    reason = _unsupported(args)
    if reason is not None:
        warning_rank_0(f"[Patch:megatron.ulysses_fused_a2a] {reason}, keeping the stock path")
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


def _has_bias(linear) -> bool:
    # TE linears without a bias may still carry an empty `bias` tensor
    bias = getattr(linear, "bias", None)
    return isinstance(bias, torch.Tensor) and bias.numel() > 0


def _input_norm(linear, x):
    """The input norm TE fuses into ``linear_qkv`` (LayerNormLinear), applied as its forward would."""
    weight = getattr(linear, "layer_norm_weight", None)
    # fused_residual_rmsnorm hands over x already normalized
    if weight is None or getattr(linear, "_skip_fused_norm", False):
        return x
    if getattr(linear, "zero_centered_gamma", False):
        weight = weight + 1
    eps = linear.eps
    if linear.normalization == "RMSNorm":
        return F.rms_norm(x, (x.shape[-1],), weight, eps)
    if linear.normalization == "LayerNorm":
        return F.layer_norm(x, (x.shape[-1],), weight, linear.layer_norm_bias, eps)
    raise RuntimeError(
        f"[Patch:megatron.ulysses_fused_a2a] unsupported normalization {linear.normalization!r}"
    )


def _run_forward_pre_hooks(module):
    # The fused path reads the module's weights without calling it, so run its forward pre-hooks the way
    # nn.Module.__call__ would; with overlap_param_gather, DDP waits there for the weights' all-gather.
    for hook_id, hook in module._forward_pre_hooks.items():
        if hook_id in module._forward_pre_hooks_with_kwargs:
            hook(module, (), {})
        else:
            hook(module, ())


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


def _make_fused_forward(args):
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
    from megatron.core.transformer.enums import AttnMaskType
    from primus_turbo.pytorch.ops import flash_attn_func
    from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import (
        ulysses_a2a_o_gemm,
        ulysses_qkv_gemm_a2a,
    )

    no_cp = _NoCPGroup()
    if _mode(args) == -1:
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
        unsupported = {
            "cross attention": key_value_states is not None,
            "inference": inference_context is not None or inference_params is not None,
            "packed sequences": packed_seq_params is not None,
            "an attention bias": attention_bias is not None,
            f"attn_mask_type={self.attn_mask_type.name}": self.attn_mask_type != AttnMaskType.causal,
            "projection biases": _has_bias(self.linear_qkv) or _has_bias(self.linear_proj),
        }
        reasons = [what for what, hit in unsupported.items() if hit]
        if reasons:
            raise NotImplementedError(
                f"[Patch:megatron.ulysses_fused_a2a] {', '.join(reasons)} not supported; the batch is already "
                "split contiguously for the fused path, so the stock attention cannot run either. "
                "Set turbo_ulysses_fused_a2a: 0"
            )

        hq, hkv, d = cfg.num_attention_heads, cfg.num_query_groups, cfg.kv_channels
        _run_forward_pre_hooks(self.linear_qkv)
        x = _input_norm(self.linear_qkv, hidden_states)
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
            softmax_scale=cfg.softmax_scale,
            causal=True,
        )
        o = o.permute(1, 0, 2, 3).contiguous()
        _run_forward_pre_hooks(self.linear_proj)
        y = o_op(o, self.linear_proj.weight, None, cp_group, "sbhd")
        return y, None

    return forward


@register_patch(
    "megatron.ulysses_fused_a2a",
    backend="megatron",
    phase="before_train",
    description="Ulysses CP through Primus-Turbo GemmA2A / A2AGemm (turbo_ulysses_fused_a2a=1, -1 serial)",
    condition=_enabled,
)
def patch_ulysses_fused_a2a(ctx: PatchContext):
    import megatron.core.models.common.embeddings.rotary_pos_embedding as rope_mod
    import megatron.core.utils as mcore_utils
    from megatron.core.transformer.attention import SelfAttention

    args = get_args(ctx)
    # imports the Turbo ops first: a missing op must fail before anything global is switched
    fused_forward = _make_fused_forward(args)
    mcore_utils._get_batch_on_this_cp_rank_per_sequence_balancing = _contiguous_cp_split
    # RoPE runs after the all-to-all over the full sequence, so keep the full-length freqs
    rope_mod.get_pos_emb_on_this_cp_rank = lambda emb, seq_dim, cp_group: emb
    SelfAttention.forward = fused_forward
    impl = "serial GEMM + all-to-all" if _mode(args) == -1 else "GemmA2A / A2AGemm"
    log_rank_0(f"[Patch:megatron.ulysses_fused_a2a] SelfAttention -> {impl}, contiguous CP split")
