###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark config fields that Megatron's ``TransformerConfig`` does not declare.

``core_transformer_config_from_args`` copies every dataclass field it finds on
``args``, so the YAML keys below need no argparse registration.
``multi_latent_attention`` must stay false, or that helper replaces this class
with ``MLATransformerConfig`` and drops these fields.

The hybrid attention pattern is derived here rather than in YAML: Primus only
evaluates string expressions for ``moe_layer_freq``, and the integer forms of
``no_rope_freq`` / ``window_attn_skip_freq`` put the special layer at the end
of each group, whereas Muse counts its groups backward from the last layer.
"""

from dataclasses import dataclass
from typing import Optional

from megatron.core.transformer.transformer_config import TransformerConfig

from primus.backends.megatron.core.models.muse_spark.muse_spark_ops import (
    global_attention_layer_mask,
)

__all__ = ["MuseSparkTransformerConfig"]


@dataclass
class MuseSparkTransformerConfig(TransformerConfig):
    """Muse Spark architecture knobs. Defaults match the released checkpoint."""

    # One full-attention NoPE layer every this many layers, counted back from the last.
    muse_global_attn_interval: int = 4
    # Sliding-window length of the RoPE layers, counted the HF way (query included).
    # ``None`` disables sliding-window attention.
    muse_sliding_window: Optional[int] = 2048
    # Q is multiplied by this after the weightless QK-RMSNorm, as HF does, and
    # softmax_scale stays 1/sqrt(head_dim): FlyDSL attention only takes that scale.
    muse_qk_scale_factor: float = 5.4730064863875
    # Applied to the router logits and to the LM-head logits.
    muse_output_multiplier: float = 0.1767766952966369
    # LM-head tanh softcap after ``muse_output_multiplier``. ``None`` disables it.
    final_logit_softcapping: Optional[float] = 20.0
    # Epsilon of the post-attention, post-MLP and post-expert norms.
    muse_post_norm_eps: float = 1.0e-8
    muse_residual_gate_temperature: float = 0.3
    muse_residual_gate_eps: float = 1.0e-3

    def __post_init__(self) -> None:
        is_global = global_attention_layer_mask(self.num_layers, self.muse_global_attn_interval)
        if self.no_rope_freq is None:
            self.no_rope_freq = [int(g) for g in is_global]
        if self.window_size is None and self.muse_sliding_window:
            # TE's left window excludes the query; HF's ``sliding_window`` includes it.
            self.window_size = (self.muse_sliding_window - 1, 0)
        elif self.window_size is not None:
            self.window_size = tuple(self.window_size)
        if self.window_size is not None and self.window_attn_skip_freq is None:
            self.window_attn_skip_freq = [int(not g) for g in is_global]

        super().__post_init__()
        self._validate_muse_spark()

    def _validate_muse_spark(self) -> None:
        required = {
            "normalization == 'RMSNorm'": self.normalization == "RMSNorm",
            "layernorm_zero_centered_gamma (set this key; apply_layernorm_1p is not read)": (
                self.layernorm_zero_centered_gamma
            ),
            "qk_l2_norm": self.qk_l2_norm,
            "not qk_layernorm": not self.qk_layernorm,
            "attention_output_gate": self.attention_output_gate,
            "not multi_latent_attention": not self.multi_latent_attention,
            "gated_linear_unit (swiglu)": self.gated_linear_unit,
            "not add_bias_linear": not self.add_bias_linear,
            "hidden_dropout == 0": self.hidden_dropout == 0.0,
            "num_moe_experts": self.num_moe_experts is not None,
            "moe_latent_size": self.moe_latent_size is not None,
            "moe_shared_expert_intermediate_size is None": self.moe_shared_expert_intermediate_size is None,
            "moe_router_score_function == 'sigmoid'": self.moe_router_score_function == "sigmoid",
            "not moe_apply_probs_on_input": not self.moe_apply_probs_on_input,
            # Until the token dispatcher reduces them, expert outputs are partial sums over
            # the expert-TP ranks, and the post-expert RMSNorm needs the full sum.
            "expert_tensor_parallel_size == 1 (it defaults to tensor_model_parallel_size)": (
                self.expert_tensor_parallel_size == 1
            ),
        }
        missing = [name for name, ok in required.items() if not ok]
        if missing:
            raise ValueError("Muse Spark requires: " + ", ".join(missing) + ".")
        if self.final_logit_softcapping is not None and self.final_logit_softcapping <= 0:
            raise ValueError(
                f"final_logit_softcapping must be positive or None, got {self.final_logit_softcapping}."
            )
