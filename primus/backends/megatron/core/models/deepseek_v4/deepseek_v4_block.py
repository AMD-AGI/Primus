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

Plan-2 P15 (this commit) — Megatron parent-class integration:

* :class:`DeepseekV4HybridLayer` now subclasses
  :class:`megatron.core.transformer.transformer_layer.TransformerLayer`
  (via ``MegatronModule`` ``__init__`` bypass) and reuses upstream
  submodule names (``input_layernorm`` / ``self_attention`` /
  ``pre_mlp_layernorm`` / ``mlp``) plus V4-specific ``attn_hc`` / ``ffn_hc``
  hooks. The submodules dataclass extends
  :class:`TransformerLayerSubmodules` so Megatron spec lifecycle code that
  inspects the layer's ``submodules_config`` works without bespoke V4
  branches.
* :class:`DeepseekV4TransformerBlock` now subclasses
  :class:`megatron.core.transformer.transformer_block.TransformerBlock`
  for type identity / sharded-state-dict integration. ``HyperHead`` is now
  built **only on the post_process stage** (it was previously built on
  every PP rank, which wasted memory and risked correctness drift).
* PP K-stream packing: ``[B, S, K, D] <-> [S*K, B, D]`` lift / lower
  helpers (:func:`_lift_streams_in` / :func:`_lower_streams_out`) carry
  the K dimension across PP P2P boundaries by folding it into the
  sequence axis. The first PP stage lifts the embedded ``[S, B, D]`` to K
  streams; intermediate stages preserve K (``[S*K, B, D]`` send/recv);
  the final stage collapses with ``HyperHead`` to ``[B, S, D]`` and
  transposes back to ``[S, B, D]`` for the output layer.
* ``token_ids`` is now a real forward kwarg threaded through
  ``DeepseekV4Model.forward -> DeepseekV4TransformerBlock.forward ->
  DeepseekV4HybridLayer.forward -> DeepseekV4MoE.forward ->
  DeepseekV4HashRouter.forward``. The legacy ``decoder._v4_token_ids``
  attribute stash is gone.
* ``position_ids`` is now consumed from the caller (forward kwarg) when
  provided; the legacy ``arange(S)`` shortcut is kept only as a fallback
  for callers that omit it (e.g. tiny CPU smokes).

Forward shape contract:
* Input ``hidden_states`` is ``[S, B, D]`` on the first PP stage and
  ``[S*K, B, D]`` on subsequent stages (where K = ``hc_mult``). The block
  reshapes to the K-stream form for HC math and packs back before
  returning to the next PP stage.
* Output is ``[S*K, B, D]`` on non-final PP stages and ``[S, B, D]`` on
  the final (post_process) stage after HyperHead collapse.
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
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import make_viewless_tensor

from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
    DeepSeekV4SpecProvider,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)
from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
    DeepseekV4Attention,
)
from primus.backends.megatron.core.transformer.dual_rope import DualRoPE
from primus.backends.megatron.core.transformer.hyper_connection import (
    HyperHead,
    HyperMixer,
)
from primus.backends.megatron.core.transformer.moe.v4_hash_router import (
    DeepseekV4HashRouter,
)
from primus.backends.megatron.core.transformer.moe.v4_moe import (
    DeepseekV4MoE,
    DeepseekV4MoESubmodules,
)
from primus.backends.megatron.core.transformer.moe.v4_topk_router import (
    DeepseekV4LearnedRouter,
)

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
    """Plain dense SwiGLU FFN with V4 pre-multiplication clamp.

    Used for non-MoE layers (or as a fallback when ``num_routed_experts``
    is 0). V4-Flash has a tiny number of dense head/tail layers; the bulk
    of layers are MoE (see :class:`DeepseekV4MoE`).

    The activation matches V4's released ``Expert.forward``:
    ``SiLU(clamp(gate, max=alpha)) * clamp(up, +/- alpha)`` with
    ``alpha = config.swiglu_limit`` (``0`` disables clamping).
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
        self.swiglu_limit = float(getattr(config, "swiglu_limit", 0.0) or 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = _projection_forward(self.w_gate, x)
        up = _projection_forward(self.w_up, x)
        if self.swiglu_limit > 0.0:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return _projection_forward(self.w_down, F.silu(gate) * up)


@dataclass
class DeepseekV4HybridLayerSubmodules(TransformerLayerSubmodules):
    """Spec tree for one DeepSeek-V4 hybrid layer.

    Plan-2 P15: extends :class:`TransformerLayerSubmodules` so the V4
    layer slots into Megatron's spec lifecycle without bespoke field
    plumbing. The four core fields use upstream-canonical names:

    * ``input_layernorm``     (was ``attn_norm``)
    * ``self_attention``      (was ``attention``)
    * ``pre_mlp_layernorm``   (was ``ffn_norm``)
    * ``mlp``                 (was ``ffn``)

    V4 adds two HC mixer hooks that the hybrid forward consumes; both
    are optional (``None`` when ``hc_mult == 1`` because there is only
    one stream to mix).

    The cross-attention / BDA fields inherited from the parent stay at
    their defaults (V4 has no cross-attention and no BDA wrapper — the
    HC residual path replaces both).
    """

    attn_hc: Optional[Union[ModuleSpec, type]] = None
    ffn_hc: Optional[Union[ModuleSpec, type]] = None


@dataclass
class DeepseekV4TransformerBlockSubmodules:
    """Spec tree for the DeepSeek-V4 decoder block.

    Plan-2 P15: ``hyper_head`` is built **only** on the post_process
    stage (the final HC collapse `[B, S, K, D] -> [B, S, D]`). On
    non-final PP stages the block keeps the K-stream form intact and
    relies on the lift / lower helpers for P2P shape compatibility.
    """

    layer_specs: Optional[List[ModuleSpec]] = None
    hyper_head: Optional[Union[ModuleSpec, type]] = None
    final_layernorm: Optional[Union[ModuleSpec, type]] = None


# ---------------------------------------------------------------------------
# K-stream <-> sequence-axis packing helpers (PP P2P shape carrier)
# ---------------------------------------------------------------------------


def _lift_streams_in(
    hidden_states: torch.Tensor,
    *,
    pre_process: bool,
    hc_mult: int,
) -> torch.Tensor:
    """Reshape the block's input to ``[B, S, K, D]`` for HC math.

    Args:
        hidden_states: incoming tensor.
            * On the first PP stage (``pre_process=True``): ``[S, B, D]``
              (Megatron's sequence-first convention) — we expand to K
              streams.
            * On subsequent PP stages (``pre_process=False``):
              ``[S*K, B, D]`` (K folded into the sequence axis by the
              previous stage's :func:`_lower_streams_out`) — we unfold.
        pre_process: ``True`` on the first PP stage.
        hc_mult: number of HC streams ``K``.

    Returns:
        ``[B, S, K, D]`` if ``hc_mult > 1`` else ``[B, S, D]``.
    """
    if hc_mult <= 1:
        # Single-stream: just transpose [S, B, D] -> [B, S, D].
        return hidden_states.transpose(0, 1).contiguous()

    if pre_process:
        # [S, B, D] -> [B, S, D] -> [B, S, K, D] (broadcast across K).
        x = hidden_states.transpose(0, 1).contiguous()
        B, S, D = x.shape
        return x.unsqueeze(2).expand(B, S, hc_mult, D).contiguous()

    # Non-first stage: [S*K, B, D] -> [B, S, K, D].
    SK, B, D = hidden_states.shape
    if SK % hc_mult != 0:
        raise ValueError(
            f"PP boundary tensor first-dim {SK} not divisible by hc_mult={hc_mult}; "
            "previous stage did not pack K via _lower_streams_out."
        )
    S = SK // hc_mult
    # [S*K, B, D] -> [S, K, B, D] -> [B, S, K, D]
    x = hidden_states.view(S, hc_mult, B, D).permute(2, 0, 1, 3).contiguous()
    return x


def _lower_streams_out(
    x: torch.Tensor,
    *,
    post_process: bool,
    hc_mult: int,
) -> torch.Tensor:
    """Reshape the block's output back to a P2P-compatible 3D tensor.

    Args:
        x: ``[B, S, K, D]`` (multi-stream) or ``[B, S, D]`` (single).
            On the final stage callers pass the post-HyperHead
            ``[B, S, D]``; on non-final stages they pass the
            still-multi-stream ``[B, S, K, D]``.
        post_process: ``True`` on the final PP stage.
        hc_mult: number of HC streams ``K``.

    Returns:
        ``[S, B, D]`` on the final stage (matching Megatron's
        sequence-first output convention) or ``[S*K, B, D]`` on
        non-final stages (K folded into the sequence axis so PP P2P
        kernels see a 3D tensor of the expected shape).
    """
    if hc_mult <= 1:
        return x.transpose(0, 1).contiguous()

    if post_process:
        # x is [B, S, D] (already collapsed by HyperHead on this stage).
        if x.dim() != 3:
            raise ValueError(
                f"_lower_streams_out: post_process expects [B, S, D] after HyperHead, "
                f"got shape {tuple(x.shape)}."
            )
        return x.transpose(0, 1).contiguous()

    # Non-final stage: pack [B, S, K, D] -> [S*K, B, D].
    if x.dim() != 4:
        raise ValueError(
            f"_lower_streams_out: non-final stage expects [B, S, K, D], " f"got shape {tuple(x.shape)}."
        )
    B, S, K, D = x.shape
    if K != hc_mult:
        raise ValueError(f"_lower_streams_out: K dim of input ({K}) does not match hc_mult ({hc_mult}).")
    # [B, S, K, D] -> [S, K, B, D] -> [S*K, B, D]
    return x.permute(1, 2, 0, 3).contiguous().view(S * K, B, D)


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
    ``attention`` :class:`ModuleSpec`.

    Plan-2 P13: all three V4 layer types (``compress_ratio in {0, 4, 128}``)
    construct through the single faithful :class:`DeepseekV4Attention`
    class, which carries its own compressor / indexer (built locally
    when no spec is provided). Production paths use
    :func:`get_deepseek_v4_runtime_decoder_spec` which provides full
    spec submodules; this fallback is for configs / unit tests that
    construct a layer without an explicit attention spec.
    """
    return DeepseekV4Attention(
        config=config,
        rope=rope,
        compress_ratio=int(compress_ratio),
    )


# ---------------------------------------------------------------------------
# A single V4 block layer (attention sub-block + FFN sub-block, both wrapped
# by HyperMixer for the K-stream residual)
# ---------------------------------------------------------------------------


class DeepseekV4HybridLayer(TransformerLayer):
    """One layer of the V4 decoder.

    Holds (using upstream-canonical submodule names):

      * :attr:`input_layernorm`     — pre-attention RMSNorm.
      * :attr:`self_attention`      — V4 attention (Dense / HCA / CSA, picked
        from ``compress_ratio``).
      * :attr:`pre_mlp_layernorm`   — pre-MLP RMSNorm.
      * :attr:`mlp`                 — V4 MoE (when ``num_moe_experts > 0``)
        or :class:`_DenseSwiGLUMLP` fallback for non-MoE layers.
      * :attr:`attn_hc` / :attr:`ffn_hc`  — :class:`HyperMixer` instances
        per sub-block when ``hc_mult > 1`` (else ``None`` and the residual
        collapses to a vanilla ``x + sub(x)`` add).

    Plan-2 P15 inherits from :class:`TransformerLayer` for type identity
    (so Megatron's ``isinstance(layer, TransformerLayer)`` checks work
    and ``BaseTransformerLayer`` -derived utilities apply). The parent
    ``__init__`` is *not* called because V4's submodule set differs from
    upstream (no cross-attention, no BDA, V4-specific attention
    signature) — we initialize via :class:`MegatronModule` directly.
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
        layer_number: Optional[int] = None,
    ) -> None:
        # Bypass TransformerLayer.__init__ (it expects the upstream
        # submodule contract — cross-attn, BDA, and a self_attention
        # signature that takes layer_number / cp_comm_type). We build
        # the V4 attribute set directly via MegatronModule.
        MegatronModule.__init__(self, config=config)

        self.layer_idx = int(layer_idx)
        self.compress_ratio = int(compress_ratio)
        self.hc_mult = int(config.hc_mult)
        # 1-based layer_number for Megatron's sharded_state_dict + recompute.
        self.layer_number = int(layer_number) if layer_number is not None else (self.layer_idx + 1)

        hidden_size = int(config.hidden_size)
        norm_eps = float(config.norm_epsilon)
        hc_eps = float(config.hc_eps)
        hc_sinkhorn_iters = int(config.hc_sinkhorn_iters)

        use_spec_submodules = submodules is not None
        submodules = submodules or DeepseekV4HybridLayerSubmodules()
        # Cache for the parent (TransformerLayer) sharded_state_dict / recompute paths.
        self.submodules_config = submodules

        if use_spec_submodules and submodules.input_layernorm is not None:
            self.input_layernorm = build_module(
                submodules.input_layernorm,
                config=config,
                hidden_size=hidden_size,
                eps=norm_eps,
            )
        else:
            self.input_layernorm = _RMSNorm(hidden_size, eps=norm_eps)

        if use_spec_submodules and submodules.self_attention is not None:
            self.self_attention = build_module(
                submodules.self_attention,
                config=config,
                rope=rope,
            )
        else:
            self.self_attention = _build_attention(
                compress_ratio=self.compress_ratio,
                rope=rope,
                config=config,
            )

        if use_spec_submodules and submodules.pre_mlp_layernorm is not None:
            self.pre_mlp_layernorm = build_module(
                submodules.pre_mlp_layernorm,
                config=config,
                hidden_size=hidden_size,
                eps=norm_eps,
            )
        else:
            self.pre_mlp_layernorm = _RMSNorm(hidden_size, eps=norm_eps)

        self.is_moe = int(config.num_moe_experts) > 0
        if use_spec_submodules and submodules.mlp is not None:
            self.mlp = build_module(
                submodules.mlp,
                config=config,
                pg_collection=pg_collection,
            )
            self.is_moe = isinstance(self.mlp, DeepseekV4MoE)
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
                hash_router=ModuleSpec(module=DeepseekV4HashRouter),
                learned_router=ModuleSpec(module=DeepseekV4LearnedRouter),
                grouped_experts=ModuleSpec(
                    module=grouped_mlp_module,
                    submodules=grouped_mlp_submodules,
                ),
                shared_expert=shared_expert_spec,
            )

            self.mlp = DeepseekV4MoE(
                config=config,
                layer_idx=self.layer_idx,
                pg_collection=pg_collection,
                submodules=moe_submodules,
            )
        else:
            self.mlp = _DenseSwiGLUMLP(
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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        position_ids: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run one V4 layer.

        Args:
            hidden_states: ``[B, S, K, D]`` (multi-stream) or
                ``[B, S, D]`` (single-stream). The block calls us with
                the K-stream form when ``hc_mult > 1`` and with the
                collapsed form otherwise.
            attention_mask: ignored — V4 manages its own SWA / sink mask
                inside :class:`DeepseekV4Attention`. Accepted for
                upstream :class:`TransformerLayer` API compatibility.
            position_ids: ``[B, S]`` or ``[S]``. Forwarded to attention.
            token_ids: ``[B, S]`` integer tensor; required when this is
                a hash-routed MoE layer
                (``layer_idx < num_hash_layers``). Ignored for non-MoE /
                non-hash layers.
            **kwargs: ignored — accepted so this layer can be placed
                inside upstream :class:`MultiTokenPredictionLayer`,
                which forwards a richer kwargs set (rotary buffers,
                inference params, etc.).
        """
        del attention_mask, kwargs

        if position_ids is None:
            # Tiny CPU smokes / unit tests may omit position_ids; fall
            # back to the seq-only arange. The block always provides
            # them in production.
            S = hidden_states.shape[1]
            position_ids = torch.arange(S, device=hidden_states.device)

        # Attention sub-block. The collapse passes a [B, S, D] hidden, then
        # the attention runs and returns [B, S, D]; HC expand writes back.
        def _attn_sub(collapsed: torch.Tensor) -> torch.Tensor:
            return self.self_attention(self.input_layernorm(collapsed), position_ids)

        x = self._hc_apply(self.attn_hc, hidden_states, _attn_sub)

        # MLP / MoE sub-block. MoE needs token_ids when the layer is
        # hash-routed; plain SwiGLU ignores it.
        if self.is_moe:

            def _ffn_sub(collapsed: torch.Tensor) -> torch.Tensor:
                return self.mlp(self.pre_mlp_layernorm(collapsed), token_ids=token_ids)

        else:

            def _ffn_sub(collapsed: torch.Tensor) -> torch.Tensor:
                return self.mlp(self.pre_mlp_layernorm(collapsed))

        x = self._hc_apply(self.ffn_hc, x, _ffn_sub)
        return x


# ---------------------------------------------------------------------------
# Top-level V4 transformer block
# ---------------------------------------------------------------------------


class DeepseekV4TransformerBlock(TransformerBlock):
    """Multi-stream HC decoder for DeepSeek-V4.

    Plan-2 P15 subclasses Megatron's
    :class:`megatron.core.transformer.transformer_block.TransformerBlock`
    for type identity (so any upstream ``isinstance(block,
    TransformerBlock)`` checks light up) and to inherit its
    sharded-state-dict / debug surface. The parent ``__init__`` is
    bypassed because V4's submodule contract differs (HyperHead on
    post-process only, K-stream lift / lower at PP boundaries, no
    upstream-style layer-norm impl override). We initialize via
    :class:`MegatronModule` directly.

    PP K-stream packing: between stages we send ``[S*K, B, D]`` (K folded
    into the sequence axis), letting standard 3D PP P2P kernels carry
    the multi-stream tensor unchanged. The first stage lifts ``[S, B, D]``
    to ``[B, S, K, D]``; the final stage collapses with HyperHead and
    transposes back to ``[S, B, D]``.

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
        # Bypass TransformerBlock.__init__: it requires a real
        # pg_collection (or pulls one from parallel_state) and runs
        # upstream-specific layer construction. V4's lift / lower path
        # plus the spec provider give us equivalent functionality with
        # CPU instantiability.
        MegatronModule.__init__(self, config=config)
        # Save arguments matching the parent's interface for compatibility.
        self.spec = spec
        self.submodules = submodules
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage
        self.pg_collection = pg_collection
        # Required by pipeline schedules (same contract as TransformerBlock).
        self.input_tensor = None
        logger.info(
            "[DeepSeek-V4] decoder block initialized (pre_process=%s post_process=%s).",
            pre_process,
            post_process,
        )

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

        # Final HC collapse: built only on the post_process stage.
        # Earlier PP stages forward the K-stream form via
        # ``_lower_streams_out`` (no HyperHead per stage).
        if hc_mult > 1 and self.post_process:
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
        position_ids: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the V4 decoder.

        Shape contract:

        * On the first PP stage (``pre_process=True``):
          ``hidden_states`` arrives as ``[S, B, D]`` (sequence-first
          embedding output). The lift helper expands it to
          ``[B, S, K, D]`` for HC math.
        * On subsequent PP stages (``pre_process=False``): the previous
          stage packed K into the sequence axis, so
          ``hidden_states`` arrives as ``[S*K, B, D]``. The lift helper
          unfolds back to ``[B, S, K, D]``.
        * On the final stage (``post_process=True``): HyperHead
          collapses ``[B, S, K, D] -> [B, S, D]``, then the lower helper
          transposes to the sequence-first ``[S, B, D]`` output.
        * On non-final stages: the lower helper packs
          ``[B, S, K, D] -> [S*K, B, D]`` so PP P2P kernels see a 3D
          tensor of the expected rank.

        ``attention_mask`` and the various ``rotary_pos_*`` kwargs are
        ignored — V4 manages its own dual-RoPE and SWA mask internally.

        ``position_ids`` is the caller-provided token-position tensor.
        When omitted (e.g. unit tests), we fall back to ``arange(S)``;
        production callers (:class:`DeepseekV4Model.forward`) always pass
        it explicitly.

        ``token_ids`` (``[B, S]`` long tensor) is required only when one
        or more layers on this stage run hash routing
        (``layer_idx < num_hash_layers``). The legacy
        ``decoder._v4_token_ids`` attribute stash has been removed; the
        model forwards ``input_ids`` here directly.
        """
        del (
            inference_context,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            packed_seq_params,
            sequence_len_offset,
            attention_mask,
            kwargs,
        )

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

        # Lift incoming P2P tensor to the K-stream form.
        x = _lift_streams_in(
            hidden_states,
            pre_process=self.pre_process,
            hc_mult=self.hc_mult,
        )
        # x is [B, S, K, D] when hc_mult > 1, else [B, S, D].
        seq_len = x.shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device)

        # Run the layers.
        for layer in self.layers:
            x = layer(
                x,
                position_ids=position_ids,
                token_ids=token_ids,
            )

        # Final HC collapse on post_process stage; non-final stages
        # forward the multi-stream form through PP P2P.
        if self.post_process and self.hc_mult > 1 and self.hyper_head is not None:
            x = self.hyper_head(x)  # [B, S, D]

        if self.final_layernorm is not None:
            x = self.final_layernorm(x)

        if not self.pre_process and len(self.layers) == 0 and self.final_layernorm is None:
            x = x.clone()

        # Lower to a P2P-compatible 3D tensor.
        out = _lower_streams_out(
            x,
            post_process=self.post_process,
            hc_mult=self.hc_mult,
        )
        return make_viewless_tensor(inp=out, requires_grad=out.requires_grad, keep_graph=True)


__all__ = [
    "DeepseekV4HybridLayerSubmodules",
    "DeepseekV4TransformerBlockSubmodules",
    "DeepseekV4HybridLayer",
    "DeepseekV4TransformerBlock",
    "_lift_streams_in",
    "_lower_streams_out",
]
