###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in GPT-OSS Q/K RMSNorm + RoPE integration.

``PRIMUS_FUSED_QK_RMSNORM_ROPE=1`` replaces the training-only sequence

    split packed QKV -> Q/K RMSNorm -> Q/K RoPE

with either Primus-Turbo's packed FlyDSL operator or AITER's two-channel RoPE
operator.  ``PRIMUS_QK_RMSNORM_ROPE_BACKEND`` selects ``flydsl`` (default) or
``aiter_2c``.  The patch is intentionally narrow: TP=1, self attention, SBHD,
non-interleaved full RoPE, no packed sequences, and PrimusTurboRMSNorm for both
Q and K.  Unsupported calls continue through the original Megatron path.
"""

from __future__ import annotations

import os

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enabled(_ctx: PatchContext) -> bool:
    value = os.environ.get("PRIMUS_FUSED_QK_RMSNORM_ROPE", "0").strip().lower()
    return value in ("1", "true", "yes", "on") and is_primus_turbo_can_patch(_ctx)


@register_patch(
    "megatron.turbo.qk_rmsnorm_rope",
    backend="megatron",
    phase="before_train",
    description="Select GPT-OSS FlyDSL fused QK path or AITER paired Q/K RoPE",
    condition=_enabled,
    priority=70,
)
def patch_qk_rmsnorm_rope(_ctx: PatchContext):
    import megatron.core.transformer.attention as attention_module
    import torch

    from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboRMSNorm

    backend = os.environ.get("PRIMUS_QK_RMSNORM_ROPE_BACKEND", "flydsl").strip().lower()
    if backend not in ("flydsl", "aiter_2c"):
        raise ValueError(
            "PRIMUS_QK_RMSNORM_ROPE_BACKEND must be 'flydsl' or 'aiter_2c'; "
            f"got {backend!r}"
        )

    if backend == "flydsl":
        from primus_turbo.pytorch.ops.rope import fused_qkv_rmsnorm_rope
    else:
        from aiter.ops.rope import rope_2c_bwd, rope_2c_fwd

        class _AiterRope2C(torch.autograd.Function):
            """Autograd bridge for AITER's paired Q/K training kernels."""

            @staticmethod
            def forward(ctx, query, key, freqs):
                if freqs.dtype != torch.float32:
                    freqs = freqs.float()
                query_out, key_out = rope_2c_fwd(query, key, freqs, 0, False, False)
                ctx.save_for_backward(freqs)
                return query_out, key_out

            @staticmethod
            def backward(ctx, grad_query, grad_key):
                (freqs,) = ctx.saved_tensors
                grad_query, grad_key = rope_2c_bwd(
                    grad_query, grad_key, freqs, 0, False, False
                )
                return grad_query, grad_key, None

    if getattr(attention_module, "_primus_qk_rmsnorm_rope_installed", False):
        return

    attention_cls = attention_module.SelfAttention
    original_forward = attention_cls.forward
    original_get_qkv = attention_cls.get_query_key_value_tensors
    original_apply_rope = attention_module.apply_rotary_pos_emb

    def _eligible(self, rotary_pos_emb, inference_context, packed_seq_params) -> bool:
        q_norm = getattr(self, "q_layernorm", None)
        k_norm = getattr(self, "k_layernorm", None)
        no_rope_freq = getattr(self.config, "no_rope_freq", None)
        no_rope = bool(no_rope_freq and no_rope_freq[self.layer_number - 1])
        return bool(
            self.training
            and inference_context is None
            and packed_seq_params is None
            and rotary_pos_emb is not None
            and not no_rope
            and self.attention_type == "self"
            and self.hidden_size_per_attention_head == 64
            and not getattr(self.config, "rotary_interleaved", False)
            and getattr(self.config, "rotary_percent", 1.0) == 1.0
            and not getattr(self.config, "attention_output_gate", False)
            and getattr(self, "world_size", 1) == 1
            and not getattr(self, "offload_qkv_linear", False)
            and isinstance(q_norm, PrimusTurboRMSNorm)
            and isinstance(k_norm, PrimusTurboRMSNorm)
            and not getattr(q_norm, "zero_centered_gamma", False)
            and not getattr(k_norm, "zero_centered_gamma", False)
            and q_norm.eps == k_norm.eps
        )

    def _forward(
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
        # Megatron's deprecated inference_params alias is resolved inside the
        # original forward.  Any inference usage is outside this training path.
        eligible = inference_params is None and _eligible(
            self, rotary_pos_emb, inference_context, packed_seq_params
        )
        if eligible:
            if not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)
            q_freqs, k_freqs = rotary_pos_emb
            # GPT-OSS uses one position table for Q and K.  Object identity is
            # deliberately required rather than paying torch.equal every layer.
            eligible = q_freqs is not None and q_freqs is k_freqs

        if eligible:
            self._primus_qk_rmsnorm_rope_context = q_freqs
        try:
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
        finally:
            if eligible:
                del self._primus_qk_rmsnorm_rope_context

    def _get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True
    ):
        freqs = getattr(self, "_primus_qk_rmsnorm_rope_context", None)
        if freqs is None or not split_qkv or output_gate:
            return original_get_qkv(
                self,
                hidden_states,
                key_value_states=key_value_states,
                output_gate=output_gate,
                split_qkv=split_qkv,
            )

        if backend == "flydsl":
            mixed_qkv, split_sizes = original_get_qkv(
                self,
                hidden_states,
                key_value_states=key_value_states,
                output_gate=False,
                split_qkv=False,
            )
            query, key, value = fused_qkv_rmsnorm_rope(
                mixed_qkv,
                self.q_layernorm.weight,
                self.k_layernorm.weight,
                freqs,
                split_sizes,
                self.q_layernorm.eps,
            )
            dispatch_shape = tuple(mixed_qkv.shape)
            dispatch_detail = f"splits={list(split_sizes)}"
        else:
            query, key, value = original_get_qkv(
                self,
                hidden_states,
                key_value_states=key_value_states,
                output_gate=False,
                split_qkv=True,
            )
            query, key = _AiterRope2C.apply(query, key, freqs)
            dispatch_shape = (tuple(query.shape), tuple(key.shape))
            dispatch_detail = "paired_qk=True"
        # The outer attention forward still visits its normal RoPE callsite.
        # Tensor-owned markers make exactly these two calls no-ops without
        # changing behavior for any other attention module or invocation.
        query._primus_qk_rmsnorm_rope_applied = True
        key._primus_qk_rmsnorm_rope_applied = True
        if not getattr(self, "_primus_qk_rmsnorm_rope_seen", False):
            self._primus_qk_rmsnorm_rope_seen = True
            log_rank_0(
                "[Patch:megatron.turbo.qk_rmsnorm_rope] First Q/K RoPE dispatch: "
                f"backend={backend} shape={dispatch_shape} {dispatch_detail}"
            )
        return query, key, value

    def _apply_rotary_pos_emb(t, *args, **kwargs):
        if getattr(t, "_primus_qk_rmsnorm_rope_applied", False):
            del t._primus_qk_rmsnorm_rope_applied
            return t
        return original_apply_rope(t, *args, **kwargs)

    attention_cls.forward = _forward
    attention_cls.get_query_key_value_tensors = _get_query_key_value_tensors
    attention_module.apply_rotary_pos_emb = _apply_rotary_pos_emb
    attention_module._primus_qk_rmsnorm_rope_installed = True
    log_rank_0(
        "[Patch:megatron.turbo.qk_rmsnorm_rope] Installed GPT-OSS Q/K RMSNorm + RoPE "
        f"backend={backend}"
    )
