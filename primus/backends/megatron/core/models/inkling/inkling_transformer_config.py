###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling (Thinking Machines) text-backbone transformer config.

Extends Megatron's :class:`TransformerConfig` with the fields Inkling needs
and upstream does not declare. The numerical reference is the official
``transformers`` implementation (``models/inkling``); field names follow its
``InklingTextConfig`` wherever Megatron has no equivalent.

Megatron fields reused as-is, with Inkling's meaning:

* ``num_attention_heads`` / ``num_query_groups`` / ``kv_channels`` -- the
  **global** (full-attention) layers. The sliding layers have their own
  ``swa_*`` geometry, which :meth:`InklingTransformerConfig.attention_config`
  swaps in per layer.
* ``ffn_hidden_size`` -- the dense MLP (``dense_intermediate_size``).
* ``moe_ffn_hidden_size`` -- one routed expert (``intermediate_size``).
* ``moe_shared_expert_intermediate_size`` -- the **total** shared width,
  ``moe_num_shared_experts * moe_ffn_hidden_size``. Megatron fuses the shared
  experts into one MLP, which is exact because the down projection is linear;
  the per-expert gammas are applied to the matching column blocks.
* ``moe_router_topk_scaling_factor`` -- ``route_scale``.
* ``moe_router_enable_expert_bias`` -- ``e_score_correction_bias``.

New fields need no argparse registration: ``train_runtime.py`` copies every
YAML key onto ``backend_args`` and ``core_transformer_config_from_args(args,
config_class=...)`` binds whatever the dataclass declares.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import List, Optional, Union

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal

__all__ = [
    "InklingTransformerConfig",
    "default_local_layer_ids",
    "normalize_moe_layer_freq",
]

_PATTERN_CHARS = re.compile(r"[^,\d\[\]\(\)\+\*\s]")

INKLING_ATTENTION_BACKENDS = ("flex", "eager")


def default_local_layer_ids(num_layers: int) -> List[int]:
    """Sliding-window layer ids when the config does not list them.

    Same rule as ``InklingTextConfig.__post_init__``: every layer whose
    1-indexed position is not a multiple of 6 slides, so the stack is five
    sliding layers then one global layer, repeated.
    """
    return [i for i in range(num_layers) if (i + 1) % 6]


def normalize_moe_layer_freq(value: Union[int, str, List[int], None], *, num_layers: int) -> List[int]:
    """Per-layer MoE pattern as ``list[int]``, ``1`` = MoE and ``0`` = dense.

    Megatron's own semantics: an int ``N`` means ``i % N == 0``, a list is
    length-checked, and the string form (``"([0]*2+[1]*40)"``, which the
    YAMLs write) is evaluated with Megatron's character whitelist.
    """
    freq = 1 if value is None else value
    if isinstance(freq, str):
        if _PATTERN_CHARS.search(freq):
            raise ValueError(f"moe_layer_freq contains unsupported characters: {freq!r}")
        freq = eval(freq, {"__builtins__": {}}, {})  # noqa: S307
    if isinstance(freq, bool):
        raise ValueError(f"Invalid moe_layer_freq: {freq!r}")
    if isinstance(freq, int):
        return [1 if (i % freq == 0) else 0 for i in range(num_layers)]
    if isinstance(freq, (list, tuple)):
        out = [int(x) for x in freq]
        if len(out) != num_layers:
            raise ValueError(f"moe_layer_freq has length {len(out)}, expected num_layers={num_layers}")
        return out
    raise ValueError(f"Invalid moe_layer_freq: {type(freq).__name__}")


@dataclass
class InklingTransformerConfig(TransformerConfig):
    """Config for the Inkling text backbone."""

    # ---- sliding-window attention ----
    swa_num_attention_heads: Optional[int] = None
    """Query heads in the sliding layers. ``None`` = ``num_attention_heads``."""

    swa_num_query_groups: Optional[int] = None
    """KV heads in the sliding layers. ``None`` = ``num_query_groups``."""

    swa_kv_channels: Optional[int] = None
    """Head dim in the sliding layers. ``None`` = ``kv_channels``."""

    sliding_window_size: int = 512
    """Window of the sliding layers; also their relative-bias extent."""

    local_layer_ids: Optional[List[int]] = None
    """0-indexed sliding layers. ``None`` = every layer but each 6th."""

    # ---- relative-position logits ----
    d_rel: int = 16
    """Per-head width of the relative states mixed into the bias."""

    rel_extent: int = 1024
    """Backward distance covered by the bias in the global layers."""

    # ---- long-context log scaling (global layers only) ----
    log_scaling_n_floor: Optional[int] = None
    """Position from which q and the bias are scaled by ``1 + a*log(n/floor)``."""

    log_scaling_alpha: float = 0.1

    # ---- short convolutions ----
    short_conv_kernel_size: int = 4
    """Depthwise causal conv on K, V, the attention output and the MLP output."""

    # ---- embedding / head ----
    use_embed_norm: bool = True
    """RMSNorm right after the token embedding."""

    logits_mup_width_multiplier: float = 16.0
    """muP divisor applied to the final hidden states before the LM head."""

    unpadded_vocab_size: Optional[int] = None
    """Rows of the LM head that hold real tokens; logits beyond are dropped."""

    # ---- MoE ----
    moe_num_shared_experts: int = 0
    """Number of shared experts, scored by the router as a softmax sink."""

    rms_norm_eps_moe_gate: float = 1e-6

    # ---- numerics ----
    inkling_fp32_activations: bool = False
    """Run the SwiGLUs (dense, routed, shared), the routing-weight products, the
    expert combine and the ``global_scale`` products in fp32, as Miles does.
    ``False`` follows the official transformers model, which keeps them in the
    model dtype."""

    # ---- kernels ----
    inkling_attention_backend: str = "eager"
    """``eager`` (dense ``[b, h, s, s]`` bias, the numerical reference) or ``flex``
    (FlexAttention score_mod and block mask; compiled, no attention dropout)."""

    def __post_init__(self):
        # The YAML string form only resolves on the launcher path (Primus's
        # moe_layer_freq args patch); configs built in-process still carry it.
        if isinstance(self.moe_layer_freq, str):
            self.moe_layer_freq = normalize_moe_layer_freq(
                self.moe_layer_freq, num_layers=int(self.num_layers)
            )
        # The reference inits every projection, output ones included, as
        # normal(0, initializer_range); Megatron would scale the output
        # projections down with depth.
        if self.output_layer_init_method is None and not self.use_mup:
            self.output_layer_init_method = init_method_normal(self.init_method_std)
        super().__post_init__()

        if self.multi_latent_attention:
            raise ValueError("Inkling does not use MLA; set multi_latent_attention: false")

        if self.swa_num_attention_heads is None:
            self.swa_num_attention_heads = self.num_attention_heads
        if self.swa_num_query_groups is None:
            self.swa_num_query_groups = self.num_query_groups
        if self.swa_kv_channels is None:
            self.swa_kv_channels = self.kv_channels
        for heads, groups, tag in (
            (self.num_attention_heads, self.num_query_groups, "global"),
            (self.swa_num_attention_heads, self.swa_num_query_groups, "sliding"),
        ):
            if heads % groups:
                raise ValueError(f"{tag} attention: {heads} heads not divisible by {groups} KV groups")

        n = int(self.num_layers)
        if self.local_layer_ids is None:
            self.local_layer_ids = default_local_layer_ids(n)
        else:
            ids = sorted({int(i) for i in self.local_layer_ids})
            if ids and (ids[0] < 0 or ids[-1] >= n):
                raise ValueError(f"local_layer_ids out of range for num_layers={n}: {ids}")
            self.local_layer_ids = ids

        if self.d_rel <= 0 or self.rel_extent <= 0 or self.sliding_window_size <= 0:
            raise ValueError("d_rel, rel_extent and sliding_window_size must be positive")
        if self.short_conv_kernel_size < 1:
            raise ValueError("short_conv_kernel_size must be >= 1")
        if self.inkling_attention_backend not in INKLING_ATTENTION_BACKENDS:
            raise ValueError(
                f"inkling_attention_backend must be one of {INKLING_ATTENTION_BACKENDS}, "
                f"got {self.inkling_attention_backend!r}"
            )

        if self.num_moe_experts:
            if self.moe_router_score_function != "sigmoid":
                raise ValueError("Inkling's router scores experts with a sigmoid")
            if self.moe_num_shared_experts < 1:
                raise ValueError("Inkling's router needs at least one shared expert (the sink)")
            shared = self.moe_num_shared_experts * int(self.moe_ffn_hidden_size)
            if self.moe_shared_expert_intermediate_size is None:
                self.moe_shared_expert_intermediate_size = shared
            elif self.moe_shared_expert_intermediate_size != shared:
                raise ValueError(
                    "moe_shared_expert_intermediate_size must equal "
                    f"moe_num_shared_experts * moe_ffn_hidden_size = {shared}, "
                    f"got {self.moe_shared_expert_intermediate_size}"
                )

    # ------------------------------------------------------------------
    # Per-layer queries (0-indexed global layer ids)
    # ------------------------------------------------------------------

    def is_sliding_layer(self, layer_idx: int) -> bool:
        return layer_idx in set(self.local_layer_ids)

    def moe_layer_pattern(self) -> List[int]:
        if not int(self.num_moe_experts or 0):
            return [0] * int(self.num_layers)
        return normalize_moe_layer_freq(self.moe_layer_freq, num_layers=int(self.num_layers))

    def is_moe_layer(self, layer_idx: int) -> bool:
        return bool(self.moe_layer_pattern()[layer_idx])

    def attention_geometry(self, layer_idx: int) -> dict:
        """Heads, KV groups, head dim, window and bias extent of one layer."""
        if self.is_sliding_layer(layer_idx):
            return dict(
                num_attention_heads=self.swa_num_attention_heads,
                num_query_groups=self.swa_num_query_groups,
                kv_channels=self.swa_kv_channels,
                window=self.sliding_window_size,
                rel_extent=self.sliding_window_size,
                sliding=True,
            )
        return dict(
            num_attention_heads=self.num_attention_heads,
            num_query_groups=self.num_query_groups,
            kv_channels=self.kv_channels,
            window=None,
            rel_extent=self.rel_extent,
            sliding=False,
        )

    def attention_config(self, layer_idx: int) -> "InklingTransformerConfig":
        """A shallow copy carrying this layer's head geometry.

        Megatron's :class:`Attention` reads the head counts off the config,
        so the sliding layers get a copy with the ``swa_*`` values swapped in.
        ``__post_init__`` is not re-run (same approach as ``SharedExpertMLP``).
        The softmax scale is ``1/head_dim``: q and k are RMS-normalised per
        head.
        """
        geo = self.attention_geometry(layer_idx)
        cfg = copy.copy(self)
        cfg.num_attention_heads = geo["num_attention_heads"]
        cfg.num_query_groups = geo["num_query_groups"]
        cfg.kv_channels = geo["kv_channels"]
        cfg.softmax_scale = 1.0 / geo["kv_channels"]
        return cfg
