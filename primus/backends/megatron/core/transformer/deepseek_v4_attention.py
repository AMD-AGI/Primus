###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 attention.

Plan-2 P13 — *faithful* attention rooted on Megatron's
``MLASelfAttention``. The dense (``compress_ratio == 0``) path
reproduces the math of the released V4-Flash checkpoint:

* Single-latent KV: a single ``linear_kv`` projection ``hidden -> head_dim``
  produces both K and V, which are broadcast across all query heads.
* Per-head ``q_rms``: a parameter-less RMS normalization on ``head_dim``
  applied AFTER ``linear_q_up_proj`` and BEFORE partial RoPE — matches
  the ``inference/model.py`` reference exactly.
* Grouped low-rank O projection: ``linear_o_a`` / ``linear_o_b`` (when
  ``config.o_lora_rank > 0``) replace the standard flat ``linear_proj``.
* Learnable per-head ``attn_sink``: an extra "virtual key" column with
  zero value, joined into the softmax. Drops the column after softmax
  so the value-weighted sum is unaffected; the head can still spend mass
  on the sink as a "no attention" fallback.
* Field names mirror MLA's canonical layout (``linear_q_down_proj``,
  ``linear_q_up_proj``, ``q_layernorm``, ``kv_layernorm``) plus the V4
  extras (``linear_kv``, ``linear_o_a``, ``linear_o_b``, ``attn_sink``)
  so the state-dict adapter (P17) can map the released safetensors keys
  (``layers.{i}.attn.{wq_a,wq_b,wkv,q_norm,kv_norm,wo_a,wo_b,attn_sink}``)
  in one straightforward table.

For now the compressed branches (``compress_ratio in {4, 128}``) continue
to ride on the plan-1 :class:`CSAAttention` / :class:`HCAAttention`
classes (which inherit from :class:`_LegacyDeepseekV4Attention` below).
The plan-2 follow-up commit folds them into this class as
``compressor`` / ``indexer`` spec submodules.

Forward signature:

.. code-block:: python

    out = attn(
        hidden,                  # [B, S, D]
        position_ids,            # [B, S] or [S]
    )
    # out: [B, S, D]
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)
from primus.backends.megatron.core.transformer.attn_sink import AttentionSink
from primus.backends.megatron.core.transformer.dual_rope import DualRoPE
from primus.backends.megatron.core.transformer.sliding_window_kv import (
    sliding_window_causal_mask,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec submodules — V4 (plan-2 / MLA-canonical)
# ---------------------------------------------------------------------------


@dataclass
class DeepseekV4AttentionSubmodules:
    """Spec submodules for the plan-2 :class:`DeepseekV4Attention`.

    The names follow MLA's canonical layout where they overlap (so that
    Megatron's standard tensor-parallel / sequence-parallel / TE machinery
    can apply unchanged), plus V4-specific extras for the single-latent KV
    and grouped low-rank O.

    Provider-built shapes:

    * ``linear_q_down_proj``  : ``hidden -> q_lora_rank``    (= ``wq_a``)
    * ``q_layernorm``        : RMSNorm on ``q_lora_rank``    (= ``q_norm``)
    * ``linear_q_up_proj``    : ``q_lora_rank -> n_heads * head_dim`` (= ``wq_b``)
    * ``linear_kv``           : ``hidden -> head_dim``       (= ``wkv``,
      single latent — broadcast to all heads)
    * ``kv_layernorm``       : RMSNorm on ``head_dim``       (= ``kv_norm``)
    * ``linear_o_a``         : ``(n_heads * head_dim / o_groups) -> o_groups * o_lora_rank``
    * ``linear_o_b``         : ``o_groups * o_lora_rank -> hidden``
    * ``attn_sink``          : :class:`AttentionSink` (per-head learnable scalar)

    When the spec provider supplies ``linear_proj`` (instead of grouped
    ``linear_o_a`` / ``linear_o_b``) the attention falls back to MLA's
    standard flat output projection — useful for unit tests and the
    ``o_lora_rank == 0`` fast-path config.
    """

    linear_q_down_proj: Optional[Union[ModuleSpec, type]] = None
    linear_q_up_proj: Optional[Union[ModuleSpec, type]] = None
    linear_kv: Optional[Union[ModuleSpec, type]] = None
    linear_o_a: Optional[Union[ModuleSpec, type]] = None
    linear_o_b: Optional[Union[ModuleSpec, type]] = None
    linear_proj: Optional[Union[ModuleSpec, type]] = None  # fallback flat O
    q_layernorm: Optional[Union[ModuleSpec, type]] = None
    kv_layernorm: Optional[Union[ModuleSpec, type]] = None
    attn_sink: Optional[Union[ModuleSpec, type]] = None


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _build_projection(
    submodule: Optional[Union[ModuleSpec, type]],
    *,
    in_features: int,
    out_features: int,
) -> nn.Module:
    """Build a linear projection from a spec submodule with fallback.

    Megatron's parallel-linear modules accept many keyword arguments that
    do not exist on plain ``nn.Linear``. If the provider-built module fails
    to construct (e.g. running outside a TP group on CPU) we fall back to
    a duplicated ``nn.Linear`` with the same shape so the unit tests can
    drive the forward pass.
    """
    if submodule is None:
        return nn.Linear(in_features, out_features, bias=False)
    try:
        return build_module(submodule)
    except Exception as exc:
        logger.warning(
            "DeepSeek-V4 attention projection submodule init failed (%s); fallback to nn.Linear.",
            exc,
        )
        return nn.Linear(in_features, out_features, bias=False)


def _projection_forward(proj: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a projection and unwrap Megatron's ``(out, bias)`` tuple."""
    out = proj(x)
    if isinstance(out, tuple):
        return out[0]
    return out


def _per_head_rms_norm(x: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Parameter-less per-head RMSNorm.

    Mirrors the released ``inference/model.py`` reference:

    .. code-block:: python

        q_rms = torch.rsqrt(q.float().square().mean(-1, keepdim=True) + eps)
        q     = (q.float() * q_rms).to(q.dtype)

    There is no learnable ``gamma`` — the per-head scale is "absorbed"
    into the surrounding ``linear_q_up_proj`` weights at training time.
    The check confirmed the released checkpoint has no separate
    ``q_rms.weight`` parameter.
    """
    in_dtype = x.dtype
    x32 = x.float()
    rsqrt = torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps)
    return (x32 * rsqrt).to(in_dtype)


# ---------------------------------------------------------------------------
# Plan-2 DeepseekV4Attention (faithful, MLA-rooted, dense-only)
# ---------------------------------------------------------------------------


class DeepseekV4Attention(MLASelfAttention):
    """V4 attention faithful to the released ``DeepSeek-V4-Flash`` checkpoint.

    Subclasses :class:`MLASelfAttention` for type identity (so downstream
    Megatron isinstance checks treat V4 attention as an MLA variant) but
    overrides ``__init__`` and ``forward`` because V4's parameter layout
    differs from MLA's compressed-KV form:

    * V4 has **no** ``linear_kv_down_proj`` / ``linear_kv_up_proj`` — the
      KV is single-latent (``wkv``) and shared as both K and V.
    * V4's ``linear_proj`` is replaced by grouped low-rank
      ``linear_o_a`` / ``linear_o_b`` (when ``config.o_lora_rank > 0``).
    * V4 adds a per-head parameter-less ``q_rms`` and a learnable
      ``attn_sink``.

    Because the parent's ``__init__`` builds modules we don't want, we
    skip the MLA / Attention init chain and call ``nn.Module.__init__``
    directly. V4-shape modules are built from the spec submodules.
    """

    def __init__(
        self,
        config: DeepSeekV4TransformerConfig,
        *,
        rope: DualRoPE,
        compress_ratio: int = 0,
        submodules: Optional[DeepseekV4AttentionSubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection=None,
    ) -> None:
        # We deliberately bypass the MLA / Attention parent __init__ chain
        # because V4's KV layout differs from MLA's compressed-KV form.
        # The class still subclasses MLASelfAttention for type identity so
        # that ``isinstance(layer.self_attention, MLASelfAttention)`` keeps
        # working in the Megatron stack.
        nn.Module.__init__(self)

        if compress_ratio != 0:
            # Plan-2 P13 first commit lands only the dense path here.
            # CSA / HCA still ride on _LegacyDeepseekV4Attention until the
            # compressor / indexer spec submodules are folded into this
            # class in a P13 follow-up.
            raise ValueError(
                f"DeepseekV4Attention currently supports compress_ratio == 0 only "
                f"(got {compress_ratio}). Use the legacy CSA / HCA classes for "
                f"compressed branches."
            )

        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        head_dim = int(config.kv_channels)
        rotary_dim = int(config.qk_pos_emb_head_dim)
        attn_sliding_window = int(config.attn_sliding_window)
        attn_sink_enabled = bool(config.attn_sink)
        attn_dropout = float(config.attention_dropout)
        norm_eps = float(getattr(config, "norm_epsilon", None) or config.layernorm_epsilon)
        q_lora_rank = int(config.q_lora_rank or 0)
        o_groups = int(getattr(config, "o_groups", 1))
        o_lora_rank = int(getattr(config, "o_lora_rank", 0))

        if q_lora_rank <= 0:
            # V4 always uses a Q LoRA path; drop the no-LoRA branch to
            # keep the math aligned with the checkpoint.
            raise ValueError(
                "DeepseekV4Attention requires config.q_lora_rank > 0; "
                "V4 always low-rank-projects Q via wq_a / wq_b."
            )

        if num_heads * head_dim % max(o_groups, 1) != 0:
            raise ValueError(
                f"num_heads * head_dim ({num_heads * head_dim}) must be divisible "
                f"by o_groups ({o_groups})"
            )

        self.config = config
        self.compress_ratio = int(compress_ratio)
        self.layer_number = int(layer_number) if layer_number is not None else 0
        self.pg_collection = pg_collection

        # ---- shape fields (read by helpers in this class) ----
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_attention_heads_per_partition = num_heads
        self.num_query_groups_per_partition = 1  # single-latent KV
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.q_head_dim = head_dim  # MLA convention; here qk_head_dim + qk_pos_emb_head_dim == head_dim
        self.attn_sliding_window = attn_sliding_window
        self.attn_dropout = attn_dropout
        self.q_lora_rank = q_lora_rank
        self.o_groups = max(o_groups, 1)
        self.o_lora_rank = o_lora_rank
        self.norm_eps = norm_eps

        # Shared dual-RoPE (held by reference; not registered to avoid
        # double-counting parameters across attention layers).
        self._rope = [rope]

        submodules = submodules or DeepseekV4AttentionSubmodules()
        self._submodules = submodules

        # ---- Q branch: hidden -> q_lora_rank -> n_heads * head_dim ----
        self.linear_q_down_proj = _build_projection(
            submodules.linear_q_down_proj,
            in_features=hidden_size,
            out_features=q_lora_rank,
        )
        if submodules.q_layernorm is None:
            self.q_layernorm = _build_local_rms_norm(q_lora_rank, eps=norm_eps)
        else:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=q_lora_rank,
                config=config,
                eps=norm_eps,
            )
        self.linear_q_up_proj = _build_projection(
            submodules.linear_q_up_proj,
            in_features=q_lora_rank,
            out_features=num_heads * head_dim,
        )

        # ---- KV branch: single-latent ``wkv`` ----
        self.linear_kv = _build_projection(
            submodules.linear_kv,
            in_features=hidden_size,
            out_features=head_dim,
        )
        if submodules.kv_layernorm is None:
            self.kv_layernorm = _build_local_rms_norm(head_dim, eps=norm_eps)
        else:
            self.kv_layernorm = build_module(
                submodules.kv_layernorm,
                hidden_size=head_dim,
                config=config,
                eps=norm_eps,
            )

        # ---- O projection ----
        # Two paths:
        #   - Grouped low-rank (V4 release): linear_o_a + linear_o_b
        #   - Flat MLA-style: linear_proj (used when o_lora_rank == 0)
        if o_lora_rank > 0:
            n_per_group = num_heads * head_dim // self.o_groups
            self.linear_o_a = _build_projection(
                submodules.linear_o_a,
                in_features=n_per_group,
                out_features=self.o_groups * o_lora_rank,
            )
            self.linear_o_b = _build_projection(
                submodules.linear_o_b,
                in_features=self.o_groups * o_lora_rank,
                out_features=hidden_size,
            )
            self.linear_proj = None
        else:
            self.linear_o_a = None
            self.linear_o_b = None
            self.linear_proj = _build_projection(
                submodules.linear_proj,
                in_features=num_heads * head_dim,
                out_features=hidden_size,
            )

        # ---- attention sink ----
        # The released checkpoint stores ``attn_sink`` as a [num_heads]
        # learnable parameter directly on the attention module
        # (key: ``layers.{i}.attn.attn_sink`` — no wrapping submodule).
        # We register it as ``self.attn_sink`` to match the checkpoint
        # key exactly, and apply softmax-with-sink inline in
        # :meth:`_attention_forward`.
        #
        # When ``submodules.attn_sink`` is supplied, the surrounding spec
        # may want a different module (e.g. a future TE-fused sink). We
        # build it in addition for forward-compat, but the canonical
        # parameter still lives on ``self.attn_sink``.
        if attn_sink_enabled:
            self.attn_sink = nn.Parameter(torch.zeros(num_heads))
        else:
            self.register_parameter("attn_sink", None)
        if attn_sink_enabled and submodules.attn_sink is not None:
            try:
                self.attn_sink_module = build_module(submodules.attn_sink, num_heads=num_heads)
            except Exception as exc:
                logger.warning(
                    "DeepSeek-V4 attn_sink submodule init failed (%s); "
                    "using inline softmax-with-sink path.",
                    exc,
                )
                self.attn_sink_module = None
        else:
            self.attn_sink_module = None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @property
    def rope(self) -> DualRoPE:
        return self._rope[0]

    def _attention_scale(self) -> float:
        base = 1.0 / math.sqrt(self.head_dim)
        rope_scale = self.rope.attn_scale(compress_ratio=self.compress_ratio)
        return base * rope_scale

    def _apply_q(self, hidden: torch.Tensor) -> torch.Tensor:
        """``[B, S, D]`` → ``[B, S, H, head_dim]`` (Q after q_norm + q_rms)."""
        q_compressed = _projection_forward(self.linear_q_down_proj, hidden)
        q_compressed = self.q_layernorm(q_compressed)
        q = _projection_forward(self.linear_q_up_proj, q_compressed)
        B, S, _ = q.shape
        q = q.view(B, S, self.num_heads, self.head_dim)
        # Per-head parameter-less RMS (matches `inference/model.py`).
        q = _per_head_rms_norm(q, eps=self.norm_eps)
        return q

    def _apply_kv(self, hidden: torch.Tensor) -> torch.Tensor:
        """``[B, S, D]`` → ``[B, S, 1, head_dim]`` (single-latent K = V)."""
        kv = _projection_forward(self.linear_kv, hidden)
        kv = self.kv_layernorm(kv)
        B, S, _ = kv.shape
        return kv.view(B, S, 1, self.head_dim)

    def _apply_rope_q_k(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        """Apply partial RoPE (last ``rotary_dim`` channels) to Q and K."""
        q = self.rope.apply_rope(q, position_ids=position_ids, compress_ratio=self.compress_ratio)
        k = self.rope.apply_rope(k, position_ids=position_ids, compress_ratio=self.compress_ratio)
        return q, k

    def _attention_forward(
        self,
        q: torch.Tensor,  # [B, H, Sq, head_dim]
        k: torch.Tensor,  # [B, H, Sk, head_dim]
        v: torch.Tensor,  # [B, H, Sk, head_dim]
        attn_mask: torch.Tensor,  # [Sq, Sk] additive
    ) -> torch.Tensor:
        """Eager scaled-dot-product attention with optional attn_sink.

        Math mirrors ``inference/model.py``: when ``self.attn_sink`` is
        non-None, a per-head learnable scalar is appended as a virtual
        key column with value zero. Softmax runs over ``[real_keys, sink]``
        but only the ``real_keys`` probabilities are used in the
        value-weighted sum. The dropped sink column lets each head spend
        attention mass on "no real key" without distorting the value sum.
        """
        scale = self._attention_scale()
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        logits = logits + attn_mask

        if self.attn_sink is not None:
            B, H, Sq, _ = logits.shape
            sink_col = self.attn_sink.float().view(1, H, 1, 1).expand(B, H, Sq, 1)
            logits_aug = torch.cat([logits, sink_col], dim=-1)
            logits_aug = logits_aug - logits_aug.amax(dim=-1, keepdim=True).detach()
            probs = logits_aug.softmax(dim=-1)[..., :-1]
        else:
            logits = logits - logits.amax(dim=-1, keepdim=True).detach()
            probs = logits.softmax(dim=-1)

        if self.attn_dropout > 0.0 and self.training:
            probs = torch.nn.functional.dropout(probs, p=self.attn_dropout)
        return torch.matmul(probs.to(v.dtype), v)

    def _grouped_o_projection(self, attn: torch.Tensor) -> torch.Tensor:
        """Apply the V4 grouped low-rank O projection.

        Input ``attn`` shape: ``[B, S, H, head_dim]``.
        Output shape: ``[B, S, hidden_size]``.

        Math (from ``inference/model.py``):

        .. code-block:: python

            # attn  : [B, S, G, (H*head_dim)/G]
            # wo_a.weight : [G * o_lora_rank, (H*head_dim)/G]
            wo_a_w = self.linear_o_a.weight.view(G, o_lora_rank, -1)
            o      = einsum("bsgd,grd->bsgr", attn, wo_a_w)
            o      = self.linear_o_b(o.flatten(2))

        We use the Linear's stored ``weight`` directly so the per-group
        einsum semantics are exact. (Megatron's parallel linears expose
        ``.weight`` after ``build_module``.)
        """
        B, S, H, Dh = attn.shape
        G = self.o_groups
        attn_g = attn.reshape(B, S, G, (H * Dh) // G)  # [B, S, G, H*Dh/G]

        wo_a = self.linear_o_a
        weight = wo_a.weight if hasattr(wo_a, "weight") else None
        if weight is None:
            # Fall back to a dense linear apply (Megatron parallel linears
            # without a directly accessible weight attribute).
            o = _projection_forward(wo_a, attn_g.reshape(B, S, -1))
            o = o.view(B, S, G * self.o_lora_rank)
        else:
            wo_a_w = weight.view(G, self.o_lora_rank, (H * Dh) // G)
            o = torch.einsum("bsgd,grd->bsgr", attn_g, wo_a_w)
            o = o.flatten(2)
        return _projection_forward(self.linear_o_b, o)

    def _flat_o_projection(self, attn: torch.Tensor) -> torch.Tensor:
        """MLA-style flat output projection (``o_lora_rank == 0`` fast path)."""
        B, S, H, Dh = attn.shape
        return _projection_forward(self.linear_proj, attn.reshape(B, S, H * Dh))

    # ------------------------------------------------------------------
    # public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """``[B, S, D] -> [B, S, D]``."""
        B, S, _ = hidden.shape
        device, dtype = hidden.device, hidden.dtype

        q = self._apply_q(hidden)  # [B, S, H, head_dim]
        kv = self._apply_kv(hidden)  # [B, S, 1, head_dim]

        # Partial RoPE on Q and K. K is post-RoPE; V uses the SAME tensor
        # (V4's single-latent design: K and V share the rope-applied kv).
        q, kv = self._apply_rope_q_k(q, kv, position_ids)
        k = kv  # [B, S, 1, head_dim]
        v = kv  # K = V (single latent)

        # Broadcast K / V across the H query-head axis.
        k_h = k.expand(B, S, self.num_heads, self.head_dim)
        v_h = v.expand(B, S, self.num_heads, self.head_dim)

        # Causal / sliding-window mask.
        window = self.attn_sliding_window if self.attn_sliding_window > 0 else 0
        if window > 0:
            attn_mask = sliding_window_causal_mask(S, window, device=device, dtype=dtype)
        else:
            attn_mask = sliding_window_causal_mask(S, S, device=device, dtype=dtype)

        # Move heads dim before sequence: [B, S, H, head_dim] -> [B, H, S, head_dim]
        q_bh = q.transpose(1, 2)
        k_bh = k_h.transpose(1, 2)
        v_bh = v_h.transpose(1, 2)

        out_bh = self._attention_forward(q_bh, k_bh, v_bh, attn_mask)
        out = out_bh.transpose(1, 2).contiguous()  # [B, S, H, head_dim]
        out = out.to(dtype=dtype)

        if self.linear_o_a is not None:
            return self._grouped_o_projection(out)
        return self._flat_o_projection(out)


def _build_local_rms_norm(dim: int, *, eps: float) -> nn.Module:
    """Tiny CPU-friendly RMSNorm used as a fallback when no spec is given."""

    class _RMSNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            in_dtype = x.dtype
            x32 = x.float()
            rsqrt = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (x32 * rsqrt).to(in_dtype) * self.weight

    return _RMSNorm()


# ---------------------------------------------------------------------------
# Legacy plan-1 attention (kept for CSA / HCA inheritance until those classes
# are folded into the plan-2 ``DeepseekV4Attention`` as spec submodules).
# ---------------------------------------------------------------------------


@dataclass
class _LegacyDeepseekV4AttentionSubmodules:
    """Plan-1 submodules dataclass.

    Retained verbatim so :class:`CSAAttention` / :class:`HCAAttention` keep
    constructing without any ``__init__`` change. The plan-2 P13 follow-up
    folds compressor / indexer into the new ``DeepseekV4Attention.forward``
    and retires this dataclass alongside the legacy class below.
    """

    linear_q_a: Optional[ModuleSpec] = None
    linear_q_b: Optional[ModuleSpec] = None
    linear_k_proj: Optional[ModuleSpec] = None
    linear_v_proj: Optional[ModuleSpec] = None
    linear_o_proj: Optional[ModuleSpec] = None


class _LegacyDeepseekV4Attention(nn.Module):
    """Plan-1 dense V4 attention (separate K / V projections, flat O).

    Continues to back the plan-1 :class:`CSAAttention` and
    :class:`HCAAttention` modules until their compressor / indexer logic
    is moved onto the plan-2 :class:`DeepseekV4Attention`.

    DO NOT use this class for new code — it does not match the released
    V4-Flash checkpoint layout (no single-latent KV, no per-head q_rms,
    no grouped low-rank O).
    """

    def __init__(
        self,
        config: DeepSeekV4TransformerConfig,
        *,
        rope: DualRoPE,
        compress_ratio: int = 0,
        submodules: Optional[_LegacyDeepseekV4AttentionSubmodules] = None,
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        num_kv_heads = int(config.num_query_groups)
        head_dim = int(config.kv_channels)
        rotary_dim = int(config.qk_pos_emb_head_dim)
        attn_sliding_window = int(config.attn_sliding_window)
        attn_sink_enabled = bool(config.attn_sink)
        q_lora_rank = config.q_lora_rank
        attn_dropout = float(config.attention_dropout)

        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.attn_sliding_window = attn_sliding_window
        self.attn_dropout = attn_dropout
        self.compress_ratio = int(compress_ratio)
        submodules = submodules or _LegacyDeepseekV4AttentionSubmodules()

        q_out = self.num_heads * self.head_dim
        if q_lora_rank is None or q_lora_rank <= 0:
            self.q_a = None
            self.q_b = _build_projection(
                submodules.linear_q_b,
                in_features=self.hidden_size,
                out_features=q_out,
            )
        else:
            self.q_a = _build_projection(
                submodules.linear_q_a,
                in_features=self.hidden_size,
                out_features=q_lora_rank,
            )
            self.q_b = _build_projection(
                submodules.linear_q_b,
                in_features=q_lora_rank,
                out_features=q_out,
            )

        kv_out = self.num_kv_heads * self.head_dim
        self.k_proj = _build_projection(
            submodules.linear_k_proj,
            in_features=self.hidden_size,
            out_features=kv_out,
        )
        self.v_proj = _build_projection(
            submodules.linear_v_proj,
            in_features=self.hidden_size,
            out_features=kv_out,
        )

        self.o_proj = _build_projection(
            submodules.linear_o_proj,
            in_features=self.num_heads * self.head_dim,
            out_features=self.hidden_size,
        )

        self._rope = [rope]

        if attn_sink_enabled:
            self.attn_sink = AttentionSink(num_heads=self.num_heads)
        else:
            self.attn_sink = None

    @property
    def rope(self) -> DualRoPE:
        return self._rope[0]

    def _project_q(self, hidden: torch.Tensor) -> torch.Tensor:
        x = _projection_forward(self.q_a, hidden) if self.q_a is not None else hidden
        q = (
            _projection_forward(self.q_b, x)
            if self.q_a is not None
            else _projection_forward(self.q_b, hidden)
        )
        B, S, _ = q.shape
        return q.view(B, S, self.num_heads, self.head_dim)

    def _project_kv(self, hidden: torch.Tensor):
        B, S, _ = hidden.shape
        k = _projection_forward(self.k_proj, hidden).view(B, S, self.num_kv_heads, self.head_dim)
        v = _projection_forward(self.v_proj, hidden).view(B, S, self.num_kv_heads, self.head_dim)
        return k, v

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        q = self.rope.apply_rope(q, position_ids=position_ids, compress_ratio=self.compress_ratio)
        k = self.rope.apply_rope(k, position_ids=position_ids, compress_ratio=self.compress_ratio)
        return q, k

    def _attention_scale(self) -> float:
        base = self.head_dim**-0.5
        rope_scale = self.rope.attn_scale(compress_ratio=self.compress_ratio)
        return base * rope_scale

    def _broadcast_kv_heads(self, kv: torch.Tensor) -> torch.Tensor:
        if self.num_kv_heads == self.num_heads:
            return kv
        repeats = self.num_heads // self.num_kv_heads
        return kv.repeat_interleave(repeats, dim=2)

    def _compute_attention_output(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        scale = self._attention_scale()
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        logits = logits + attn_mask

        if self.attn_sink is not None:
            return self.attn_sink(logits, v, dropout=self.attn_dropout)

        probs = logits.softmax(dim=-1)
        if self.attn_dropout > 0.0 and self.training:
            probs = torch.nn.functional.dropout(probs, p=self.attn_dropout)
        probs = probs.to(v.dtype)
        return torch.matmul(probs, v)

    def _extra_kv(self, hidden, position_ids, q):
        return None, None, None

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, S, _ = hidden.shape
        device, dtype = hidden.device, hidden.dtype

        q = self._project_q(hidden)
        k_local, v_local = self._project_kv(hidden)
        q, k_local = self._apply_rope(q, k_local, position_ids)

        k_local_h = self._broadcast_kv_heads(k_local)
        v_local_h = self._broadcast_kv_heads(v_local)

        window = self.attn_sliding_window
        local_mask = sliding_window_causal_mask(S, window, device=device, dtype=dtype)

        extra_k, extra_v, extra_mask = self._extra_kv(hidden, position_ids, q)

        if extra_k is not None:
            k_full = torch.cat([k_local_h, extra_k], dim=1)
            v_full = torch.cat([v_local_h, extra_v], dim=1)
            full_mask = torch.cat([local_mask, extra_mask], dim=-1)
        else:
            k_full = k_local_h
            v_full = v_local_h
            full_mask = local_mask

        q_bh = q.transpose(1, 2)
        k_bh = k_full.transpose(1, 2)
        v_bh = v_full.transpose(1, 2)

        out_bh = self._compute_attention_output(q_bh, k_bh, v_bh, full_mask)
        out = out_bh.transpose(1, 2).contiguous()
        out = out.to(dtype=dtype)

        out = out.reshape(B, S, self.num_heads * self.head_dim)
        out = _projection_forward(self.o_proj, out)
        return out


__all__ = [
    "DeepseekV4Attention",
    "DeepseekV4AttentionSubmodules",
    "_LegacyDeepseekV4Attention",
    "_LegacyDeepseekV4AttentionSubmodules",
]
