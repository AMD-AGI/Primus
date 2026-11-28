###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark gated self-attention.

Megatron's ``SelfAttention`` with ``attention_output_gate=True`` already packs
the sigmoid gate per query group in ``linear_qkv`` and applies
``x * sigmoid(gate)`` after core attention. Muse Spark adds a weightless
per-head RMSNorm (``o_norm``) on the core-attention output before the gate.
The weightless QK-norm comes in through ``q_layernorm`` / ``k_layernorm`` in
the layer spec (:class:`MuseSparkQKNorm`); the query scale
``qk_scale_factor`` is folded into ``softmax_scale`` by the config.

With ``muse_spark_fused_qk_norm_rope`` the split of the gated packed QKV, the
weightless Q / K norms and the rotate-half RoPE (none on NoPE layers) run as
Primus-Turbo's ``fused_qkv_rmsnorm_rope`` (gate block, no norm weights; its Triton
backend), one kernel per pass. The norm output and the inverse-RoPE gradient are
rounded to bf16 as the unfused modules do; the RoPE is rounded once per output from
fp32 cos / sin (as TE's fused RoPE), not after each multiply and add as Megatron's
unfused RoPE. The forward stashes the RoPE angles and runs Megatron's forward
without them; ``get_query_key_value_tensors`` applies them.
"""

import torch
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.transformer.attention import SelfAttention
from megatron.core.typed_torch import apply_module

from primus.backends.megatron.core.models.muse_spark.model_types import config_flag
from primus.backends.megatron.core.transformer.muse_spark.compiled_norms import (
    compiled,
    o_norm_gate_math,
)
from primus.backends.megatron.core.transformer.muse_spark.norms import (
    MuseSparkRMSNorm,
    weightless_rms,
)
from primus.core.utils.module_utils import log_rank_0

__all__ = ["MuseSparkQKNorm", "MuseSparkSelfAttention"]


def _turbo_shape_error(qkv, freqs, split):
    """Why Primus-Turbo's ``fused_qkv_rmsnorm_rope`` cannot take this input (gate block, no weights), or None.
    A Turbo without that op, or whose op still requires the norm weights, gives a reason, not an error."""
    try:
        from primus_turbo.pytorch.kernels.rope.qk_rmsnorm_rope_impl import (
            qk_rmsnorm_rope_shape_error,
        )

        return qk_rmsnorm_rope_shape_error(qkv, None, None, freqs, split)
    except (ImportError, AttributeError, TypeError) as exc:
        return f"Primus-Turbo has no fused_qkv_rmsnorm_rope with a gate block ({exc})"


class MuseSparkQKNorm(MuseSparkRMSNorm):
    """Weightless per-head RMSNorm for the q / k heads: ``MuseSparkRMSNorm(with_scale=False)``.

    Only fixes ``with_scale=False`` (Megatron builds q / k layernorms with
    ``(config=, hidden_size=, eps=)``); the fp32 math is MuseSparkRMSNorm's.
    """

    def __init__(self, config=None, hidden_size: int = None, eps: float = 1e-5, **kw):
        kw.pop("with_scale", None)
        super().__init__(config, hidden_size, eps=eps, with_scale=False, **kw)


class MuseSparkSelfAttention(SelfAttention):
    """SelfAttention with a weightless per-head ``o_norm`` before the sigmoid output gate."""

    def __init__(self, config, *args, **kwargs):
        assert config.attention_output_gate, "Muse Spark attention needs attention_output_gate=True"
        super().__init__(config, *args, **kwargs)
        self.o_norm_eps = float(config.layernorm_epsilon)
        self.o_norm_fused = config_flag(config, "muse_spark_fused_attn_norm", True)
        self.o_norm_compile = config_flag(config, "muse_spark_compile_norms", False)
        self.qk_norm_rope_fused = config_flag(config, "muse_spark_fused_qk_norm_rope", False)
        self._qk_norm_rope = None

    def _qk_norm_rope_context(self, args, kwargs):
        """``(freqs or None, rope)`` when this call can take the fused Q / K norm + RoPE, else None."""
        if not (self.qk_norm_rope_fused and not args and self.attention_type == "self"):
            return None
        if any(kwargs.get(k) is not None for k in (
            "key_value_states", "inference_context", "inference_params", "packed_seq_params",
            "rotary_pos_cos", "rotary_pos_sin", "rotary_pos_cos_sin",
        )):  # fmt: skip
            return None
        cfg = self.config
        cp = getattr(getattr(self, "pg_collection", None), "cp", None)
        if (
            cfg.test_mode
            or getattr(self, "world_size", 1) != 1
            or (cp is not None and cp.size() != 1)
            or not isinstance(self.q_layernorm, MuseSparkQKNorm)
            or not isinstance(self.k_layernorm, MuseSparkQKNorm)
            or self.q_layernorm.eps != self.k_layernorm.eps
        ):
            return None
        if cfg.no_rope_freq and cfg.no_rope_freq[self.layer_number - 1]:
            return None, False
        freqs = kwargs.get("rotary_pos_emb")
        if isinstance(freqs, tuple):
            freqs = freqs[0] if freqs[0] is freqs[1] else None
        if (
            freqs is None
            or cfg.rotary_interleaved
            or cfg.apply_rope_fusion
            or getattr(self, "_yarn_concentration_factor", 1.0) != 1.0
            or freqs.dim() != 4
            or freqs.shape[-1] != self.hidden_size_per_attention_head
        ):
            return None
        return freqs, True

    def forward(self, hidden_states, attention_mask, *args, **kwargs):
        ctx = self._qk_norm_rope_context(args, kwargs)
        if ctx is None:
            return super().forward(hidden_states, attention_mask, *args, **kwargs)
        # Megatron then applies no RoPE; get_query_key_value_tensors does, fused with the norms
        kwargs["rotary_pos_emb"] = None
        self._qk_norm_rope = ctx
        try:
            return super().forward(hidden_states, attention_mask, **kwargs)
        finally:
            self._qk_norm_rope = None

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True
    ):
        ctx = self._qk_norm_rope
        if ctx is None or not output_gate or not split_qkv or key_value_states is not None:
            return super().get_query_key_value_tensors(
                hidden_states, key_value_states, output_gate=output_gate, split_qkv=split_qkv
            )
        freqs, rope = ctx
        mixed_qkv, _ = apply_module(self.linear_qkv)(hidden_states)
        hn = self.hidden_size_per_attention_head
        heads = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        mixed_qkv = mixed_qkv.view(*mixed_qkv.shape[:2], self.num_query_groups_per_partition, -1)
        split = [heads * hn, heads * hn, hn, hn]
        # the same freqs object every layer, so Turbo builds its cos / sin tables once
        if rope and freqs.shape[0] != mixed_qkv.shape[0]:
            freqs = freqs[: mixed_qkv.shape[0]]
        rope_freqs = freqs if rope else None
        if _turbo_shape_error(mixed_qkv, rope_freqs, split) is not None:
            return self._split_norm_rope(mixed_qkv, freqs, rope, heads, hn)
        from primus_turbo.pytorch.ops.rope import fused_qkv_rmsnorm_rope

        query, key, value, gate = fused_qkv_rmsnorm_rope(
            mixed_qkv, None, None, rope_freqs, split, self.q_layernorm.eps
        )
        if not self.__dict__.get("_qk_norm_rope_logged"):
            self._qk_norm_rope_logged = True
            log_rank_0(
                f"[Primus:muse_spark] layer {self.layer_number}: fused gated QKV + Q/K norm "
                f"{'+ RoPE ' if rope else '(NoPE) '}shape={tuple(mixed_qkv.shape)}"
            )
        return query, key, value, gate.reshape(*gate.shape[:2], -1, hn)

    def _split_norm_rope(self, mixed_qkv, freqs, rope, heads, hn):
        """Megatron's unfused split, Q / K norms and RoPE, for an input the fused kernel does not take."""
        query, gate, key, value = torch.split(mixed_qkv, [heads * hn, heads * hn, hn, hn], dim=3)
        query = apply_module(self.q_layernorm)(query.reshape(*query.shape[:2], -1, hn))
        key = apply_module(self.k_layernorm)(key)
        if rope:
            freqs = freqs[: mixed_qkv.shape[0]]
            cp = self.pg_collection.cp
            query = apply_rotary_pos_emb(query, freqs, config=self.config, cp_group=cp)
            key = apply_rotary_pos_emb(key, freqs, config=self.config, cp_group=cp)
        return query, key, value, gate.reshape(*gate.shape[:2], -1, hn)

    def _apply_output_gate(self, x, gate):
        # x: [sq, b, np * hn] (or [t, 1, np * hn] for thd); normalize each head.
        hn = self.hidden_size_per_attention_head
        if self.o_norm_compile and x.is_cuda:
            return compiled(o_norm_gate_math)(x, gate, hn, self.o_norm_eps)
        shape = x.shape
        # fp32 per-head norm, rounded to x.dtype; the fused kernel on GPU (weightless_rms)
        x = weightless_rms(x.view(*shape[:-1], -1, hn), self.o_norm_eps, self.o_norm_fused).view(shape)
        return super()._apply_output_gate(x, gate)
