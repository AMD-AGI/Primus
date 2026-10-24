###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling transformer config.

Extends Megatron's :class:`TransformerConfig` with the fields the Inkling
modules need and that upstream does not declare. The fields are bound from
the YAML by ``core_transformer_config_from_args(args, config_class=...)``;
no argparse registration is needed.

Inkling (HF ``inkling_mm_model`` text backbone) per decoder layer:

* attention without RoPE: q/k RMSNorm per head, depthwise causal short
  convolutions on k and v, a learned relative-position logit bias
  (``d_rel`` x ``rel_extent`` profile bank mixed per token and head),
  scores scaled by ``1 / head_dim``;
* layers alternate local (sliding window, ``inkling_swa_num_query_groups``
  KV heads, bias extent = window) and global (``num_query_groups`` KV heads,
  bias extent ``inkling_rel_extent``);
* a short convolution on the attention output and on the MLP output;
* a sigmoid top-k MoE whose router also scores the shared experts
  ("shared-expert sink"), renormalised over the chosen + shared experts.

``multi_latent_attention`` must stay False: ``core_transformer_config_from_args``
would otherwise replace this class by ``MLATransformerConfig``.
"""

from dataclasses import dataclass
from typing import Optional

from megatron.core.transformer.transformer_config import TransformerConfig

__all__ = ["InklingTransformerConfig"]


@dataclass
class InklingTransformerConfig(TransformerConfig):
    """Inkling-specific fields (defaults: the released Inkling text backbone)."""

    inkling_local_layer_period: int = 2
    """Layer ``i`` is a local (sliding-window) layer when ``i % period == 0``; the others are global.
    HF ``local_layer_ids`` [0, 2, 4, ...]."""

    inkling_sliding_window: int = 512
    """Sliding window of the local layers (a key at distance ``q - k`` in ``[0, window)`` is visible).
    It is also the relative-bias extent of the local layers."""

    inkling_swa_num_query_groups: int = 4
    """KV heads of the local layers (the global layers use ``num_query_groups``)."""

    inkling_rel_dim: int = 16
    """Width of the per-head relative states (HF ``d_rel``)."""

    inkling_rel_extent: int = 1024
    """Relative-bias extent of the global layers (HF ``rel_extent``)."""

    inkling_sconv_kernel_size: int = 4
    """Kernel width of the depthwise causal short convolutions."""

    inkling_num_routed_experts: int = 256
    inkling_moe_router_topk: int = 6
    inkling_num_shared_experts: int = 2
    inkling_route_scale: float = 8.0

    inkling_logits_mup_divisor: float = 3.0
    """The final hidden state is divided by this before the output layer (HF logits_mup_width_multiplier)."""

    inkling_unpadded_vocab_size: Optional[int] = None
    """When set, logits of ids >= this are masked out (HF unpadded_vocab_size). Left unset for training."""

    inkling_log_scaling_n_floor: Optional[int] = 128000
    inkling_log_scaling_alpha: float = 0.1
    """Global layers scale queries and the relative bias by 1 + alpha * log(max(1, (pos + 1) / n_floor))."""

    def __post_init__(self):
        super().__post_init__()
        assert self.inkling_local_layer_period >= 1
        assert self.num_attention_heads % self.inkling_swa_num_query_groups == 0
