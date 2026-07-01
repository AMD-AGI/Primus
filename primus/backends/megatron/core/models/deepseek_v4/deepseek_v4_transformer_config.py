###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 specific transformer config.

This config extends Megatron's ``MLATransformerConfig`` with DeepSeek-V4
runtime fields that are referenced by V4 modules but are not part of the
upstream ``TransformerConfig``/``MLATransformerConfig`` schema.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from megatron.core.transformer.transformer_config import MLATransformerConfig

# ``mtp_compress_ratios`` and ``v4_use_custom_mtp_block`` lived here in
# plan-1 / plan-2 P12-P16 as escape hatches for the legacy primus-owned
# :class:`DeepseekV4MTPBlock`. Plan-2 P17 retired that block (the MTP path
# is now exclusively the spec-based upstream
# :class:`MultiTokenPredictionBlock` route via
# :func:`get_v4_mtp_block_spec`); both fields are deliberately gone here
# and their references in YAML configs / training scripts must be removed.


def _normalize_compress_ratios_field(
    value: Optional[Union[str, List[int], Tuple[int, ...]]],
    *,
    field_name: str = "compress_ratios",
) -> Optional[Tuple[int, ...]]:
    """Plan-2 P18 (D4 audit): normalize ``compress_ratios`` to a tuple.

    YAML loaders deliver this field as a *string* (e.g.
    ``"[0, 0, 4, 128, ...]"``) when wrapped in quotes, or as an actual
    list when written without quotes. The dataclass stored both as
    ``Optional[Union[str, List[int], Tuple[int, ...]]]``, and runtime
    helpers (``_parse_int_sequence`` / ``_normalize_compress_ratios``
    in ``deepseek_v4_block.py``) had to ``ast.literal_eval`` the string
    on every consumer path.

    With this helper running once in ``__post_init__``, every consumer
    sees a single canonical type — ``tuple[int, ...]`` — and the runtime
    parsing layer is reduced to a length-fitting check.

    Args:
        value: raw config value (string, list, tuple, or ``None``).
        field_name: only used for error messages.

    Returns:
        ``None`` (when ``value`` is ``None``) or a tuple of ints.
    """
    if value is None:
        return None
    parsed = value
    if isinstance(parsed, str):
        try:
            parsed = ast.literal_eval(parsed)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a list-like value, got {value!r}") from exc
    if isinstance(parsed, (list, tuple)):
        try:
            return tuple(int(x) for x in parsed)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} entries must be int-castable; got {parsed!r}") from exc
    raise TypeError(f"{field_name} must be a list/tuple/str, got {type(parsed).__name__}")


@dataclass
class DeepSeekV4TransformerConfig(MLATransformerConfig):
    # ---- DeepSeek-V4 hybrid attention / HC ----
    hc_mult: int = 1
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1.0e-6

    # ---- DeepSeek-V4 MTP (multi-token prediction) ----
    # The released V4 checkpoint gives each MTP depth its OWN small
    # ``hc_head_fn`` (a per-depth :class:`HyperHead`) instead of reusing the
    # main trunk's HyperHead. With ``hc_mult > 1`` the MTP inner
    # :class:`DeepseekV4HybridLayer` runs on the K-stream form and must be
    # collapsed back to a single stream by this per-depth head before the
    # MTP final-layernorm + shared LM head. Set ``False`` to fall back to a
    # single-stream MTP inner layer (no per-depth head); see
    # ``deepseek_v4_mtp_layer.DeepseekV4MTPLayer``.
    mtp_use_separate_hc_head: bool = True

    compress_ratios: Optional[Union[str, List[int], Tuple[int, ...]]] = None
    compress_rope_theta: float = 160000.0

    # ---- DeepSeek-V4 attention extras ----
    attn_sliding_window: int = 0
    attn_sink: bool = False
    index_topk: int = 0
    index_head_dim: int = 128
    index_n_heads: int = 64

    # ---- DeepSeek-V4 FP8 Indexer (CSA selector QK path) ----
    # When True, the CSA :class:`Indexer` fake-quantizes its query / compressed
    # key activations to FP8 (E4M3) before the QK scoring einsum, simulating
    # the released V4 low-precision indexer QK path (the report runs the
    # indexer QK in FP4/FP8 while keeping the BF16 index-score / top-k path).
    # The score reduction (ReLU + per-head weight + sum + causal mask) and the
    # top-k selection stay in the activation dtype (BF16). The indexer is a
    # non-differentiable, frozen top-k selector, so this is an inference-side
    # precision reduction (no straight-through estimator needed). Default False
    # so existing BF16 runs are bit-unchanged; enable via
    # ``PRIMUS_USE_V4_FP8_INDEXER`` (see run_deepseek_v4.sh / flash yaml).
    use_v4_fp8_indexer: bool = False

    # ---- DeepSeek-V4 attention backend selection (unified string selectors) ----
    # ``use_v4_attention_backend`` selects the dense (cr=0) / HCA (cr=128) kernel;
    # ``use_v4_csa_attention_backend`` selects the CSA (cr=4) kernel:
    #   dense/HCA: eager | triton_v1 | triton_v2 | gluon
    #   CSA:       eager | triton_v0 | triton_v1 | triton_v2 | gluon | flydsl_v0
    # (triton_v0 = deprecated gathered; flydsl_v0 = deprecated legacy FlyDSL,
    #  fwd-only). ``use_turbo_attention`` (when a ``core_attention`` module is
    # built) still takes precedence for the dense path.
    use_v4_attention_backend: str = "gluon"
    use_v4_csa_attention_backend: str = "gluon"

    # ---- Per-module precision (paper recipe): routed experts in MXFP4 while the
    # rest of the layer runs FP8. Decoupled from the global --fp4/--fp8 recipe
    # (which are mutually exclusive): with FP8 on, the PrimusTurbo grouped MLP
    # routes the expert GEMMs through the native FP4 (hipBLASLt) path. Default OFF.
    moe_experts_fp4: bool = False

    # ---- DeepSeek-V4 plan-5 P29: torch.compile-fused Sinkhorn ----
    # Plan-5 P29 (RESCOPED from "small-op fusion" — see plan-5 02-phase-
    # details.md). The Sinkhorn-Knopp doubly-stochastic projection inside
    # ``HyperMixer.compute_weights`` (see ``hyper_connection.py``) issues
    # 1 + 2 * (n_iters - 1) separate fp32 ``aten::sum`` reductions per
    # call. At V4-Flash production widths the per-call shape is
    # ``[B, S, K, K] = [1, 4096, 4, 4]`` and HIP's default
    # ``reduce_kernel<512, 1, ...>`` runs the kernel ~250x over the
    # memory-bound floor. The P28 baseline trace pinned this kernel as
    # 87.3 % of step time. Setting this flag to ``True`` collapses every
    # such call into a single ``torch.compile(fullgraph=True,
    # dynamic=False)`` Inductor-fused Triton kernel; AOT autograd handles
    # the BWD. Default ``False`` until G32 (FWD + BWD parity) and G33b
    # (post-P29 trace) flip it on.
    use_v4_compiled_sinkhorn: bool = False

    # ---- DeepSeek-V4 grouped low-rank output projection ----
    # Mirrors the released checkpoint's `wo_a` / `wo_b` layout.
    # When ``o_lora_rank == 0`` the attention falls back to a flat O proj
    # (Megatron's ``linear_proj``); set it >0 to use the grouped low-rank
    # form (``linear_o_a`` + ``linear_o_b``) with ``o_groups`` groups.
    o_groups: int = 8
    o_lora_rank: int = 0

    # ---- DeepSeek-V4 MoE routing / expert extras ----
    num_hash_layers: int = 0
    hash_routing_seed: int = 0

    moe_intermediate_size: Optional[int] = None
    moe_use_legacy_grouped_gemm: bool = False

    swiglu_limit: float = 0.0
    v4_grouped_experts_support_clamped_swiglu: bool = False

    # ---- Vocab helpers used by hash router ----
    vocab_size: Optional[int] = None
    padded_vocab_size: Optional[int] = None

    # ---- Compat aliases for V4 code paths ----
    norm_epsilon: Optional[float] = None
    position_embedding_type: str = "none"

    def __post_init__(self) -> None:
        # P18 D4: normalize compress_ratios once so downstream helpers
        # always see ``tuple[int, ...]`` (or None). YAML strings like
        # ``"[0, 0, 4, ...]"`` are evaluated here.
        if self.compress_ratios is not None:
            self.compress_ratios = _normalize_compress_ratios_field(
                self.compress_ratios, field_name="compress_ratios"
            )

        # Keep V4's ``norm_epsilon`` alias consistent with MCore's
        # ``layernorm_epsilon`` before parent validation runs.
        if self.norm_epsilon is None:
            self.norm_epsilon = float(self.layernorm_epsilon)
        self.layernorm_epsilon = float(self.norm_epsilon)

        # DeepSeek naming compatibility for MoE hidden size.
        if self.moe_ffn_hidden_size is None and self.moe_intermediate_size is not None:
            self.moe_ffn_hidden_size = int(self.moe_intermediate_size)

        # Keep DeepSeek clamp name aligned with MCore clamp field.
        clamp_from_activation = self.activation_func_clamp_value
        clamp_from_swiglu = float(self.swiglu_limit)
        if clamp_from_activation is None:
            if clamp_from_swiglu > 0.0:
                self.activation_func_clamp_value = clamp_from_swiglu
        elif clamp_from_swiglu <= 0.0:
            self.swiglu_limit = float(clamp_from_activation)

        # Ensure hash-router vocab lookups always have a concrete size.
        if self.padded_vocab_size is None and self.vocab_size is not None:
            self.padded_vocab_size = int(self.vocab_size)
        if self.vocab_size is None and self.padded_vocab_size is not None:
            self.vocab_size = int(self.padded_vocab_size)

        super().__post_init__()

        if self.moe_intermediate_size is None and self.moe_ffn_hidden_size is not None:
            self.moe_intermediate_size = int(self.moe_ffn_hidden_size)


__all__ = ["DeepSeekV4TransformerConfig"]
