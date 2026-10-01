###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in GPT-OSS packed-QKV RMSNorm + RoPE integration.

``PRIMUS_FUSED_QK_RMSNORM_ROPE=1`` replaces the training-only sequence

    split packed QKV -> Q/K RMSNorm -> Q/K RoPE

with Primus-Turbo's packed FlyDSL operator.  The patch is intentionally narrow:
TP=1, self attention, SBHD, non-interleaved full RoPE, no packed sequences,
and PrimusTurboRMSNorm for both Q and K.  Unsupported calls continue through
the original Megatron path.
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
    description="Fuse GPT-OSS packed QKV split, Q/K RMSNorm, and RoPE with FlyDSL",
    condition=_enabled,
    priority=70,
)
def patch_qk_rmsnorm_rope(_ctx: PatchContext):
    import megatron.core.transformer.attention as attention_module
    from primus_turbo.pytorch.kernels.rope.qk_rmsnorm_rope_impl import (
        qk_rmsnorm_rope_shape_error,
    )
    from primus_turbo.pytorch.ops.rope import fused_qkv_rmsnorm_rope

    from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboRMSNorm

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
            and attention_module._yarn_get_concentration_factor_from_config(self.config) == 1.0
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

        mixed_qkv, split_sizes = original_get_qkv(
            self,
            hidden_states,
            key_value_states=key_value_states,
            output_gate=False,
            split_qkv=False,
        )
        why = qk_rmsnorm_rope_shape_error(
            mixed_qkv,
            self.q_layernorm.weight,
            self.k_layernorm.weight,
            freqs,
            split_sizes,
        )
        if why is not None:
            if attention_module.SplitAlongDim is not None:
                query, key, value = attention_module.SplitAlongDim(mixed_qkv, 3, split_sizes)
            else:
                query, key, value = mixed_qkv.split(split_sizes, dim=3)
            query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
            query = attention_module.apply_module(self.q_layernorm)(query)
            key = attention_module.apply_module(self.k_layernorm)(key)
            if self.config.test_mode:
                self.run_realtime_tests()
            return query, key, value
        query, key, value = fused_qkv_rmsnorm_rope(
            mixed_qkv,
            self.q_layernorm.weight,
            self.k_layernorm.weight,
            freqs,
            split_sizes,
            self.q_layernorm.eps,
        )
        # The outer attention forward still visits its normal RoPE callsite.
        # Tensor-owned markers make exactly these two calls no-ops without
        # changing behavior for any other attention module or invocation.
        query._primus_qk_rmsnorm_rope_applied = True
        key._primus_qk_rmsnorm_rope_applied = True
        if not getattr(self, "_primus_qk_rmsnorm_rope_seen", False):
            self._primus_qk_rmsnorm_rope_seen = True
            log_rank_0(
                "[Patch:megatron.turbo.qk_rmsnorm_rope] First fused packed-QKV dispatch: "
                f"shape={tuple(mixed_qkv.shape)} splits={list(split_sizes)}"
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
        "[Patch:megatron.turbo.qk_rmsnorm_rope] Installed GPT-OSS FlyDSL "
        "packed QKV + Q/K RMSNorm + RoPE fusion"
    )
