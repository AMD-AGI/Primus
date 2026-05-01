###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 transformer block (multi-stream HC + per-layer attention dispatch).

Reference:
* techblog §1 ("Hybrid Attention") — per-layer attention selected by
  ``compress_ratios[layer_id]``.
* techblog §2 ("mHC: Manifold-Constrained Hyper-Connections") — ``hc_mult``
  parallel hidden streams, mixed via per-layer ``HyperMixer`` and final
  ``HyperHead`` collapse.

Phase 6 contract:
* This is a **standalone** ``nn.Module``. It does not inherit from
  ``megatron.core.transformer.transformer_block.TransformerBlock``; instead
  it exposes the same call signature so :class:`DeepseekV4Model` can
  ``self.decoder = DeepseekV4TransformerBlock(...)`` and Megatron's
  ``GPTModel.forward`` keeps working.
* PP local-layer partitioning is wired via Megatron's layer-offset helpers
  (build only this rank's decoder layers + accept ``set_input_tensor``).
* The FFN sub-block supports both dense SwiGLU and V4 MoE (hash-routed
  prefix + learned router tail).

Forward shape contract:
* Input ``hidden_states`` is ``[S, B, D]`` (Megatron's sequence-first
  convention). The block transposes to ``[B, S, D]`` internally, runs the
  K-stream HC loop, and transposes back before return.
* Output is ``[S, B, D]`` of the same dtype/device.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import make_viewless_tensor

from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
    DeepSeekV4SpecProvider,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)
from primus.backends.megatron.core.transformer.csa_attention import CSAAttention
from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
    DeepseekV4Attention,
    _LegacyDeepseekV4Attention,
)
from primus.backends.megatron.core.transformer.dual_rope import DualRoPE
from primus.backends.megatron.core.transformer.hca_attention import HCAAttention
from primus.backends.megatron.core.transformer.hyper_connection import (
    HyperHead,
    HyperMixer,
)
from primus.backends.megatron.core.transformer.moe.v4_hash_router import HashRouter
from primus.backends.megatron.core.transformer.moe.v4_moe import (
    DeepseekV4MoE,
    DeepseekV4MoESubmodules,
)
from primus.backends.megatron.core.transformer.moe.v4_topk_router import V4TopKRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider-aware projection helpers
# ---------------------------------------------------------------------------


def _default_init_method(_weight: torch.Tensor) -> None:
    return None


def _build_projection(
    in_features: int,
    out_features: int,
    *,
    config: DeepSeekV4TransformerConfig,
) -> nn.Module:
    if config is None:
        return nn.Linear(in_features, out_features, bias=False)

    provider = DeepSeekV4SpecProvider(config=config)
    linear_module_cls = provider.linear()
    init_method: Callable = config.init_method or _default_init_method
    try:
        return linear_module_cls(
            input_size=in_features,
            output_size=out_features,
            parallel_mode="duplicated",
            config=config,
            init_method=init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            tp_comm_buffer_name=None,
            is_expert=False,
        )
    except Exception as exc:
        logger.warning(
            "DeepSeek-V4 MLP projection provider linear init failed (%s); fallback to nn.Linear.",
            exc,
        )
        return nn.Linear(in_features, out_features, bias=False)


def _projection_forward(proj: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = proj(x)
    if isinstance(out, tuple):
        return out[0]
    return out


# ---------------------------------------------------------------------------
# Pieces used by every layer
# ---------------------------------------------------------------------------


def _parse_int_sequence(value, *, field_name: str) -> Optional[List[int]]:
    """Parse config-provided sequence fields into ``List[int]``.

    YAML values may arrive as:
    - actual list/tuple: ``[0, 4, 128, ...]``
    - stringified list: ``"[0, 4, 128, ...]"``
    """
    if value is None:
        return None

    parsed = value
    if isinstance(parsed, str):
        try:
            parsed = ast.literal_eval(parsed)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a list-like value, got {value!r}") from exc

    if isinstance(parsed, torch.Tensor):
        parsed = parsed.tolist()

    if not isinstance(parsed, (list, tuple)):
        raise TypeError(f"{field_name} must be list/tuple, got {type(parsed).__name__}")

    out: List[int] = []
    for i, item in enumerate(parsed):
        try:
            out.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}[{i}]={item!r} is not int-castable") from exc
    return out


def _normalize_compress_ratios(
    compress_ratios,
    *,
    num_layers: int,
    mtp_num_layers: int,
) -> List[int]:
    """Normalize ``compress_ratios`` to exactly ``num_layers`` entries."""
    ratios = _parse_int_sequence(compress_ratios, field_name="compress_ratios")
    if ratios is None:
        return [0] * num_layers

    if len(ratios) == num_layers:
        return ratios

    # Common DeepSeek layout: decoder ratios + mtp ratios in one list.
    if len(ratios) == num_layers + mtp_num_layers:
        logger.warning(
            "compress_ratios has decoder+MTP length (%s), truncating to decoder num_layers (%s).",
            len(ratios),
            num_layers,
        )
        return ratios[:num_layers]

    if len(ratios) > num_layers:
        logger.warning(
            "compress_ratios length (%s) > num_layers (%s), truncating.",
            len(ratios),
            num_layers,
        )
        return ratios[:num_layers]

    # len(ratios) < num_layers: extend with last ratio (or 0 if empty).
    pad_value = ratios[-1] if ratios else 0
    logger.warning(
        "compress_ratios length (%s) < num_layers (%s), padding with %s.",
        len(ratios),
        num_layers,
        pad_value,
    )
    return ratios + [pad_value] * (num_layers - len(ratios))


class _RMSNorm(nn.Module):
    """Standalone RMSNorm so the block has no hard dep on TE."""

    def __init__(
        self,
        dim: Optional[int] = None,
        eps: float = 1e-6,
        hidden_size: Optional[int] = None,
        config=None,
    ) -> None:
        del config
        super().__init__()
        if dim is None:
            dim = hidden_size
        if dim is None:
            raise ValueError("RMSNorm requires `dim` or `hidden_size`.")
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        rsqrt = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * rsqrt).to(in_dtype) * self.weight


class _DenseSwiGLUMLP(nn.Module):
    """Plain dense SwiGLU FFN.

    Used for non-MoE layers (or as a fallback when ``num_routed_experts``
    is 0). V4-Flash has a tiny number of dense head/tail layers; the bulk
    of layers are MoE (see :class:`DeepseekV4MoE`).
    """

    def __init__(
        self,
        config: DeepSeekV4TransformerConfig,
        *,
        pg_collection=None,
    ) -> None:
        del pg_collection
        super().__init__()
        hidden_size = int(config.hidden_size)
        ffn_hidden_size = int(config.ffn_hidden_size)
        self.w_gate = _build_projection(
            hidden_size,
            ffn_hidden_size,
            config=config,
        )
        self.w_up = _build_projection(
            hidden_size,
            ffn_hidden_size,
            config=config,
        )
        self.w_down = _build_projection(
            ffn_hidden_size,
            hidden_size,
            config=config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = _projection_forward(self.w_gate, x)
        up = _projection_forward(self.w_up, x)
        return _projection_forward(self.w_down, F.silu(gate) * up)


@dataclass
class DeepseekV4HybridLayerSubmodules:
    """Spec tree for one DeepSeek-V4 hybrid layer."""

    attn_norm: Union[ModuleSpec, type] = _RMSNorm
    attention: Union[ModuleSpec, type] = DeepseekV4Attention
    ffn_norm: Union[ModuleSpec, type] = _RMSNorm
    ffn: Union[ModuleSpec, type] = _DenseSwiGLUMLP
    attn_hc: Optional[Union[ModuleSpec, type]] = None
    ffn_hc: Optional[Union[ModuleSpec, type]] = None


@dataclass
class DeepseekV4TransformerBlockSubmodules:
    """Spec tree for the DeepSeek-V4 decoder block."""

    layer_specs: Optional[List[ModuleSpec]] = None
    hyper_head: Optional[Union[ModuleSpec, type]] = None
    final_layernorm: Optional[Union[ModuleSpec, type]] = None


# ---------------------------------------------------------------------------
# Per-layer attention factory
# ---------------------------------------------------------------------------


def _build_attention(
    *,
    compress_ratio: int,
    rope: DualRoPE,
    config: Optional[DeepSeekV4TransformerConfig] = None,
):
    """No-spec fallback used when the layer is built without an
    ``attention`` :class:`ModuleSpec`. Production paths construct the
    attention via :func:`get_deepseek_v4_runtime_decoder_spec`, which
    routes ``compress_ratio == 0`` through the new V4-faithful
    :class:`DeepseekV4Attention`. This fallback keeps the legacy
    plan-1 path alive for unit tests / configs that pre-date the
    P13 rewrite.

    * ``0``   → :class:`_LegacyDeepseekV4Attention` (dense + SWA + sink)
    * ``128`` (or any larger ratio: HCA convention) → :class:`HCAAttention`
    * ``4``   → :class:`CSAAttention` (overlap compressor + Indexer)
    """
    common = dict(rope=rope, config=config)

    if compress_ratio == 0:
        return _LegacyDeepseekV4Attention(compress_ratio=0, **common)
    if compress_ratio == 4:
        return CSAAttention(
            compress_ratio=compress_ratio,
            **common,
        )
    return HCAAttention(
        compress_ratio=compress_ratio,
        **common,
    )


# ---------------------------------------------------------------------------
# A single V4 block layer (attention sub-block + FFN sub-block, both wrapped
# by HyperMixer for the K-stream residual)
# ---------------------------------------------------------------------------


class DeepseekV4HybridLayer(GraphableMegatronModule):
    """One layer of the V4 decoder.

    Holds:
      * pre-attention RMSNorm
      * attention sub-block (Dense / HCA / CSA, picked from ``compress_ratio``)
      * pre-FFN RMSNorm
      * FFN sub-block: MoE (:class:`DeepseekV4MoE`) when ``num_routed_experts > 0``,
        otherwise a plain dense SwiGLU. The MoE handles its own router dispatch
        (hash for the first ``num_hash_layers`` layers, sqrtsoftplus / sigmoid
        / softmax for the rest).
      * Two :class:`HyperMixer` instances (one per sub-block) when ``hc_mult > 1``
    """

    def __init__(
        self,
        config: DeepSeekV4TransformerConfig,
        *,
        layer_idx: int,
        compress_ratio: int,
        rope: DualRoPE,
        pg_collection=None,
        submodules: Optional[DeepseekV4HybridLayerSubmodules] = None,
    ) -> None:
        super().__init__(config)
        self.layer_idx = int(layer_idx)
        self.compress_ratio = int(compress_ratio)
        self.hc_mult = int(config.hc_mult)

        hidden_size = int(config.hidden_size)
        norm_eps = float(config.norm_epsilon)
        hc_eps = float(config.hc_eps)
        hc_sinkhorn_iters = int(config.hc_sinkhorn_iters)

        use_spec_submodules = submodules is not None
        submodules = submodules or DeepseekV4HybridLayerSubmodules()

        if use_spec_submodules and submodules.attn_norm is not None:
            self.attn_norm = build_module(
                submodules.attn_norm,
                config=config,
                hidden_size=hidden_size,
                eps=norm_eps,
            )
        else:
            self.attn_norm = _RMSNorm(hidden_size, eps=norm_eps)

        if use_spec_submodules and submodules.attention is not None:
            self.attn = build_module(
                submodules.attention,
                config=config,
                rope=rope,
            )
        else:
            self.attn = _build_attention(
                compress_ratio=self.compress_ratio,
                rope=rope,
                config=config,
            )

        if use_spec_submodules and submodules.ffn_norm is not None:
            self.ffn_norm = build_module(
                submodules.ffn_norm,
                config=config,
                hidden_size=hidden_size,
                eps=norm_eps,
            )
        else:
            self.ffn_norm = _RMSNorm(hidden_size, eps=norm_eps)

        self.is_moe = int(config.num_moe_experts) > 0
        if use_spec_submodules and submodules.ffn is not None:
            self.ffn = build_module(
                submodules.ffn,
                config=config,
                pg_collection=pg_collection,
            )
            self.is_moe = isinstance(self.ffn, DeepseekV4MoE)
        elif self.is_moe:
            moe_use_grouped_gemm = bool(config.moe_grouped_gemm)
            moe_use_legacy_grouped_gemm = bool(config.moe_use_legacy_grouped_gemm)

            provider = DeepSeekV4SpecProvider(config=config)
            grouped_mlp_module, grouped_mlp_submodules = provider.v4_grouped_mlp_modules(
                moe_use_grouped_gemm=moe_use_grouped_gemm,
                moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
            )
            assert (
                grouped_mlp_module is not None
            ), "DeepSeek-V4 grouped MLP module must be provided by DeepSeekV4SpecProvider."

            shared_expert_spec = ModuleSpec(
                module=SharedExpertMLP,
                submodules=MLPSubmodules(
                    linear_fc1=provider.column_parallel_linear(),
                    linear_fc2=provider.row_parallel_linear(),
                    activation_func=provider.activation_func(),
                ),
            )
            moe_submodules = DeepseekV4MoESubmodules(
                hash_router=ModuleSpec(module=HashRouter),
                learned_router=ModuleSpec(module=V4TopKRouter),
                grouped_experts=ModuleSpec(
                    module=grouped_mlp_module,
                    submodules=grouped_mlp_submodules,
                ),
                shared_expert=shared_expert_spec,
            )

            self.ffn = DeepseekV4MoE(
                config=config,
                layer_idx=self.layer_idx,
                pg_collection=pg_collection,
                submodules=moe_submodules,
            )
        else:
            self.ffn = _DenseSwiGLUMLP(
                config=config,
            )

        if self.hc_mult > 1:
            if use_spec_submodules and submodules.attn_hc is not None:
                self.attn_hc = build_module(
                    submodules.attn_hc,
                    hidden_size=hidden_size,
                    hc_mult=self.hc_mult,
                    eps=hc_eps,
                    sinkhorn_iters=hc_sinkhorn_iters,
                )
            else:
                self.attn_hc = HyperMixer(
                    hidden_size=hidden_size,
                    hc_mult=self.hc_mult,
                    eps=hc_eps,
                    sinkhorn_iters=hc_sinkhorn_iters,
                )
            if use_spec_submodules and submodules.ffn_hc is not None:
                self.ffn_hc = build_module(
                    submodules.ffn_hc,
                    hidden_size=hidden_size,
                    hc_mult=self.hc_mult,
                    eps=hc_eps,
                    sinkhorn_iters=hc_sinkhorn_iters,
                )
            else:
                self.ffn_hc = HyperMixer(
                    hidden_size=hidden_size,
                    hc_mult=self.hc_mult,
                    eps=hc_eps,
                    sinkhorn_iters=hc_sinkhorn_iters,
                )
        else:
            self.attn_hc = None
            self.ffn_hc = None

    # ------------------------------------------------------------------

    def _hc_apply(self, mixer: Optional[HyperMixer], x: torch.Tensor, sub_block, *args):
        """Run a sub-block under HC.

        ``x`` shape: ``[B, S, K, D]`` if ``hc_mult > 1``, else ``[B, S, D]``.
        ``sub_block`` is ``Callable[[Tensor, *Any], Tensor]`` whose first
        positional arg is the (collapsed) hidden in ``[B, S, D]``.
        """
        if mixer is None:
            # Single-stream: classic residual; x already has shape [B, S, D].
            out = sub_block(x, *args)
            return x + out

        pre, post, comb = mixer.compute_weights(x)  # [..., K], [..., K], [..., K, K]
        collapsed = HyperMixer.collapse(x, pre)  # [B, S, D]
        out = sub_block(collapsed, *args)  # sub-block first positional = collapsed
        return HyperMixer.expand(x, out, post, comb)  # [B, S, K, D]

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one V4 layer.

        ``x``: ``[B, S, K, D]`` (multi-stream) or ``[B, S, D]`` (single).
        ``position_ids``: ``[B, S]`` or ``[S]``.
        ``token_ids``: ``[B, S]`` integer tensor; required when this is a
            hash-routed MoE layer (``layer_idx < num_hash_layers``). Ignored
            for non-MoE / non-hash layers.
        """

        # Attention sub-block. The collapse passes a [B, S, D] hidden, then
        # the attention runs and returns [B, S, D]; HC expand writes back.
        def _attn_sub(collapsed: torch.Tensor) -> torch.Tensor:
            return self.attn(self.attn_norm(collapsed), position_ids)

        x = self._hc_apply(self.attn_hc, x, _attn_sub)

        # FFN sub-block. MoE FFN needs token_ids when the layer is hash-routed;
        # plain SwiGLU ignores it.
        if self.is_moe:

            def _ffn_sub(collapsed: torch.Tensor) -> torch.Tensor:
                return self.ffn(self.ffn_norm(collapsed), token_ids=token_ids)

        else:

            def _ffn_sub(collapsed: torch.Tensor) -> torch.Tensor:
                return self.ffn(self.ffn_norm(collapsed))

        x = self._hc_apply(self.ffn_hc, x, _ffn_sub)
        return x


# ---------------------------------------------------------------------------
# Top-level V4 transformer block
# ---------------------------------------------------------------------------


class DeepseekV4TransformerBlock(nn.Module):
    """Multi-stream HC decoder for DeepSeek-V4.

    Replaces Megatron's ``TransformerBlock`` for V4. The ``__init__``
    signature accepts a :class:`DeepSeekV4TransformerConfig` object so it can be
    constructed exactly the way ``GPTModel.__init__`` constructs the
    standard block.

    Phase 6 update:
    - respects PP / VP layer partitioning by constructing only local layers
      for this pipeline rank;
    - supports ``set_input_tensor`` so non-first PP stages consume P2P input.
    """

    def __init__(
        self,
        config: DeepSeekV4TransformerConfig,
        spec=None,
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection=None,
        vp_stage=None,
        submodules: Optional[DeepseekV4TransformerBlockSubmodules] = None,
    ) -> None:
        super().__init__()
        # Save arguments matching the parent's interface for compatibility.
        self.config = config
        self.spec = spec
        self.submodules = submodules
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage
        # Required by pipeline schedules (same contract as TransformerBlock).
        self.input_tensor = None
        logger.info("[DeepSeek-V4] decoder block initialized.")

        # ---- shape / model fields ----
        hidden_size = config.hidden_size
        rotary_dim = config.qk_pos_emb_head_dim
        num_layers = config.num_layers
        norm_eps = config.norm_epsilon

        # ---- V4-specific fields ----
        hc_mult = config.hc_mult
        hc_eps = config.hc_eps
        config.hc_sinkhorn_iters
        compress_ratios = _normalize_compress_ratios(
            config.compress_ratios,
            num_layers=num_layers,
            mtp_num_layers=int(config.mtp_num_layers),
        )
        self.compress_ratios: List[int] = compress_ratios

        rope_theta = config.rotary_base
        compress_rope_theta = config.compress_rope_theta
        yarn_factor = config.rotary_scaling_factor
        original_max_pos = config.original_max_position_embeddings

        self.num_hash_layers = int(config.num_hash_layers)

        # ---- shared dual-RoPE for the whole stack ----
        self.rope = DualRoPE(
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            compress_rope_theta=compress_rope_theta,
            yarn_factor=yarn_factor,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            original_max_position_embeddings=original_max_pos,
        )

        # ---- stage-local layer specs (always provided by runtime spec) ----
        provided_layer_specs = submodules.layer_specs if submodules is not None else None
        assert provided_layer_specs, "DeepSeek-V4 requires non-empty submodules.layer_specs."
        self.layers = nn.ModuleList()
        self.global_layer_indices = []
        for local_idx, layer_spec in enumerate(provided_layer_specs):
            layer = build_module(
                layer_spec,
                config=config,
                pg_collection=pg_collection,
                rope=self.rope,
            )
            self.layers.append(layer)
            self.global_layer_indices.append(int(getattr(layer, "layer_idx", local_idx)))
        self.layer_offset = self.global_layer_indices[0] if self.global_layer_indices else 0
        self.hc_mult = hc_mult

        # Final HC collapse (only if multi-stream).
        if hc_mult > 1:
            if submodules is not None and submodules.hyper_head is not None:
                self.hyper_head = build_module(
                    submodules.hyper_head,
                    hidden_size=hidden_size,
                    hc_mult=hc_mult,
                    eps=hc_eps,
                )
            else:
                self.hyper_head = HyperHead(hidden_size=hidden_size, hc_mult=hc_mult, eps=hc_eps)
        else:
            self.hyper_head = None

        # Final RMSNorm placement follows Megatron semantics:
        # - no MTP: on post_process stage
        # - with MTP: on the stage containing decoder's final layer
        if self._has_final_layernorm_in_this_stage(total_decoder_layers=num_layers):
            if submodules is not None and submodules.final_layernorm is not None:
                self.final_layernorm = build_module(
                    submodules.final_layernorm,
                    config=self.config,
                    hidden_size=hidden_size,
                    eps=norm_eps,
                )
            else:
                self.final_layernorm = _RMSNorm(hidden_size, eps=norm_eps)
        else:
            self.final_layernorm = None

    # ------------------------------------------------------------------

    def _has_final_layernorm_in_this_stage(self, *, total_decoder_layers: int) -> bool:
        if not self.post_layer_norm:
            return False

        mtp_num_layers = self.config.mtp_num_layers
        if mtp_num_layers is None:
            return self.post_process

        if not self.global_layer_indices:
            return False
        return self.global_layer_indices[-1] == (total_decoder_layers - 1)

    def set_input_tensor(self, input_tensor: torch.Tensor):
        """Pipeline-parallel hook: stash tensor from previous PP stage."""
        self.input_tensor = input_tensor

    @property
    def num_layers_per_pipeline_rank(self) -> int:
        """Compatibility shim used by upstream debug / recompute code."""
        return len(self.layers)

    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the V4 decoder.

        Megatron passes ``hidden_states`` as ``[S, B, D]`` (sequence-first).
        We transpose to ``[B, S, D]`` for HC math and back for the return.
        ``attention_mask`` and the various ``rotary_pos_*`` kwargs are
        ignored — V4 manages its own dual-RoPE and SWA mask internally.

        ``token_ids`` (``[B, S]`` long tensor) is required only when one
        or more layers run hash routing (``layer_idx < num_hash_layers``).
        :class:`DeepseekV4Model` forwards ``input_ids`` here for that
        purpose; non-V4 callers (e.g. probes / unit tests of layers
        without hash routing) can omit it.
        """
        # If the model stashed ``input_ids`` on the block (see
        # :class:`DeepseekV4Model.forward`), pick it up; explicit kwarg wins.
        if token_ids is None:
            token_ids = getattr(self, "_v4_token_ids", None)

        if not self.pre_process:
            hidden_states = self.input_tensor if self.input_tensor is not None else hidden_states
        if hidden_states is None:
            raise ValueError("DeepseekV4TransformerBlock.forward received no hidden_states tensor")

        needs_hash_token_ids = self.num_hash_layers > 0 and any(
            layer_idx < self.num_hash_layers for layer_idx in self.global_layer_indices
        )
        if needs_hash_token_ids and token_ids is None:
            raise ValueError(
                "token_ids is required on this PP stage because it owns hash-routed MoE layers "
                f"(global layer idx < num_hash_layers={self.num_hash_layers})."
            )

        # [S, B, D] -> [B, S, D]
        x = hidden_states.transpose(0, 1).contiguous()
        B, S, D = x.shape

        # Position ids (one per token).
        position_ids = torch.arange(S, device=x.device)

        # Expand to K streams.
        if self.hc_mult > 1:
            x = x.unsqueeze(2).expand(B, S, self.hc_mult, D).contiguous()

        # Run the layers.
        for layer in self.layers:
            x = layer(x, position_ids, token_ids=token_ids)

        # Final HC collapse.
        if self.hc_mult > 1 and self.hyper_head is not None:
            x = self.hyper_head(x)  # [B, S, D]

        if self.final_layernorm is not None:
            x = self.final_layernorm(x)

        if not self.pre_process and len(self.layers) == 0 and self.final_layernorm is None:
            x = x.clone()

        # Back to [S, B, D] for downstream Megatron code.
        out = x.transpose(0, 1).contiguous()
        return make_viewless_tensor(inp=out, requires_grad=out.requires_grad, keep_graph=True)


__all__ = [
    "DeepseekV4HybridLayerSubmodules",
    "DeepseekV4TransformerBlockSubmodules",
    "DeepseekV4HybridLayer",
    "DeepseekV4TransformerBlock",
]
