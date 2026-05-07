###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 attention.

Plan-2 P13 — *faithful* attention rooted on Megatron's
``MLASelfAttention``. The released ``DeepSeek-V4-Flash`` checkpoint is
reproduced for **all three** layer types (``compress_ratio in {0, 4, 128}``)
inside a single attention class:

* Single-latent KV: a single ``linear_kv`` projection ``hidden -> head_dim``
  produces both K and V, broadcast across all query heads.
* Per-head ``q_rms``: a parameter-less RMS normalization on ``head_dim``
  applied AFTER ``linear_q_up_proj`` and BEFORE partial RoPE — matches
  the ``inference/model.py`` reference exactly.
* Grouped low-rank O projection: ``linear_o_a`` / ``linear_o_b`` (when
  ``config.o_lora_rank > 0``) replace the standard flat ``linear_proj``.
* Learnable per-head ``attn_sink``: an extra "virtual key" column with
  zero value, joined into the softmax. Drops the column after softmax
  so the value-weighted sum is unaffected; the head can still spend mass
  on the sink as a "no attention" fallback.
* Compressed branches (``compress_ratio > 0``) fold their compressor
  (and indexer for CSA) in as :class:`Compressor` / :class:`Indexer`
  spec submodules; the dense local SWA branch and the compressed branch
  are softmax-joined together so the attention sink is shared across
  both paths.
* Field names mirror MLA's canonical layout (``linear_q_down_proj``,
  ``linear_q_up_proj``, ``q_layernorm``, ``kv_layernorm``) plus the V4
  extras (``linear_kv``, ``linear_o_a``, ``linear_o_b``, ``compressor``,
  ``indexer``) so the state-dict adapter (P17) can map the released
  safetensors keys
  (``layers.{i}.attn.{wq_a,wq_b,wkv,q_norm,kv_norm,wo_a,wo_b,attn_sink,
  compressor.*,indexer.*}``) in one straightforward table.  The
  per-head learnable softmax sink lives directly on the attention module
  as ``self.attn_sink: nn.Parameter`` (no submodule slot — Plan-3 P21
  dropped the dead ``attn_sink`` field; the inline softmax-with-sink
  path in :meth:`_attention_forward` is canonical).

Forward signature:

.. code-block:: python

    out = attn(
        hidden,                  # [B, S, D]
        position_ids,            # [B, S] or [S]
    )
    # out: [B, S, D]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)
from primus.backends.megatron.core.transformer.compressor import Compressor
from primus.backends.megatron.core.transformer.dual_rope import (
    DualRoPE,
    apply_interleaved_partial_rope,
)
from primus.backends.megatron.core.transformer.indexer import Indexer
from primus.backends.megatron.core.transformer.local_rmsnorm import LocalRMSNorm
from primus.backends.megatron.core.transformer.sliding_window_kv import (
    sliding_window_causal_mask,
)


_SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)


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
    * ``compressor``         : :class:`Compressor` (compress_ratio > 0 only)
    * ``indexer``            : :class:`Indexer`    (compress_ratio == 4 only)

    When the spec provider supplies ``linear_proj`` (instead of grouped
    ``linear_o_a`` / ``linear_o_b``) the attention falls back to MLA's
    standard flat output projection — useful for unit tests and the
    ``o_lora_rank == 0`` fast-path config.

    Plan-3 P21: there is no ``attn_sink`` submodule slot.  The per-head
    learnable sink is :class:`torch.nn.Parameter` ``self.attn_sink``
    on the attention module itself (key ``layers.{i}.attn.attn_sink``
    in the released checkpoint), and the softmax-with-sink combine is
    inlined in :meth:`DeepseekV4Attention._attention_forward`.
    """

    linear_q_down_proj: Optional[Union[ModuleSpec, type]] = None
    linear_q_up_proj: Optional[Union[ModuleSpec, type]] = None
    linear_kv: Optional[Union[ModuleSpec, type]] = None
    linear_o_a: Optional[Union[ModuleSpec, type]] = None
    linear_o_b: Optional[Union[ModuleSpec, type]] = None
    linear_proj: Optional[Union[ModuleSpec, type]] = None  # fallback flat O
    q_layernorm: Optional[Union[ModuleSpec, type]] = None
    kv_layernorm: Optional[Union[ModuleSpec, type]] = None
    compressor: Optional[Union[ModuleSpec, type]] = None
    indexer: Optional[Union[ModuleSpec, type]] = None


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _build_projection(
    submodule: Optional[Union[ModuleSpec, type]],
    *,
    in_features: int,
    out_features: int,
) -> nn.Module:
    """Build a linear projection from a spec submodule.

    When the spec is ``None`` (CPU unit tests that exercise the
    forward pass without a TP group) we instantiate a plain
    :class:`nn.Linear` with the same shape.  When a spec is supplied
    we delegate to :func:`build_module` and let any constructor
    failure bubble up — Plan-3 P21 retired the ``try/except/return
    nn.Linear`` fallback because it produced an unsharded model
    (vanilla ``nn.Linear`` instead of column / row parallel shards)
    that silently masked spec bugs at TP=1 and would diverge at TP>1.
    """
    if submodule is None:
        return nn.Linear(in_features, out_features, bias=False)
    return build_module(submodule)


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


def _build_local_rms_norm(dim: int, *, eps: float) -> nn.Module:
    """Tiny CPU-friendly RMSNorm used as a fallback when no spec is given.

    Plan-2 P17 retired the closure-built ``_RMSNorm`` helper here; the
    canonical implementation lives in
    :class:`primus.backends.megatron.core.transformer.local_rmsnorm.LocalRMSNorm`
    so the same code path is shared with
    :mod:`...deepseek_v4_block` and :mod:`...compressor`.
    """
    return LocalRMSNorm(dim=dim, eps=eps)


# ---------------------------------------------------------------------------
# DeepseekV4Attention (faithful, MLA-rooted, dense + CSA + HCA)
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
    * V4 layers come in three flavours selected by ``compress_ratio``:

      * ``0``   — dense / SWA over local KV.
      * ``128`` — HCA: local SWA *plus* a fully-visible compressed pool
        (Compressor in non-overlap mode).
      * ``4``   — CSA: local SWA *plus* a per-query top-K selection over
        a compressed pool (Compressor in overlap mode + Indexer).

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
        attn_mask_type=None,
        **kwargs,
    ) -> None:
        # We deliberately bypass the MLA / Attention parent __init__ chain
        # because V4's KV layout differs from MLA's compressed-KV form.
        # The class still subclasses MLASelfAttention for type identity so
        # that ``isinstance(layer.self_attention, MLASelfAttention)`` keeps
        # working in the Megatron stack.
        #
        # Plan-2 P16: ``attn_mask_type`` is accepted (and ignored) so the
        # attention spec can declare a value that satisfies upstream
        # :class:`MultiTokenPredictionLayer`'s pre-build validator; V4
        # manages its own SWA / sink mask internally. ``**kwargs`` swallows
        # any forward-compatible kwargs upstream may add (e.g.
        # ``cp_comm_type``) so the spec lifecycle keeps working.
        del attn_mask_type, kwargs
        nn.Module.__init__(self)

        if compress_ratio not in _SUPPORTED_COMPRESS_RATIOS:
            raise ValueError(
                f"DeepseekV4Attention supports compress_ratio in "
                f"{_SUPPORTED_COMPRESS_RATIOS} (got {compress_ratio})."
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
        # learnable parameter directly on the attention module (key
        # ``layers.{i}.attn.attn_sink`` — no wrapping submodule).
        # We register it as ``self.attn_sink`` so the state-dict key
        # matches the released checkpoint exactly; the softmax-with-sink
        # combine is inlined in :meth:`_attention_forward`.
        #
        # Plan-3 P21 retired the optional ``self.attn_sink_module``
        # build branch (and the ``submodules.attn_sink`` slot) — the
        # branch was never exercised in the forward path and its
        # ``try/except`` masked AttentionSink build failures.  A future
        # TE-fused sink primitive can land as a new spec field once it
        # actually replaces the inline path.
        if attn_sink_enabled:
            self.attn_sink = nn.Parameter(torch.zeros(num_heads))
        else:
            self.register_parameter("attn_sink", None)

        # ---- compressor / indexer (compressed branches only) ----
        self.compressor: Optional[nn.Module] = None
        self.indexer: Optional[nn.Module] = None
        if self.compress_ratio > 0:
            self.compressor = self._build_compressor(submodules.compressor)
            if self.compress_ratio == 4:
                self.indexer = self._build_indexer(submodules.indexer)

    # ------------------------------------------------------------------
    # construction helpers (compressed branches)
    # ------------------------------------------------------------------

    def _build_compressor(self, spec: Optional[Union[ModuleSpec, type]]) -> nn.Module:
        """Build the V4 :class:`Compressor` for compressed branches.

        Plan-1 conventions (kept under V4): ``ratio=4`` → overlap mode
        (CSA), ``ratio=128`` → non-overlap mode (HCA). The released
        checkpoint hard-codes ``coff=2`` for overlap (CSA) and ``coff=1``
        for non-overlap (HCA); :class:`Compressor` enforces this through
        its own ``overlap`` argument.

        When the spec is ``None`` (CPU unit tests, ``DeepseekV4Attention``
        constructed without a layer spec) we instantiate the local
        :class:`Compressor` directly.  Otherwise we delegate to
        :func:`build_module` and let any constructor failure bubble up —
        Plan-3 P21 retired the ``try/except/local Compressor`` fallback
        because the spec passes the same :class:`Compressor` class and
        the fallback handler was dead code that masked real spec bugs.
        """
        kwargs = dict(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            ratio=self.compress_ratio,
            overlap=(self.compress_ratio == 4),
        )
        if spec is None:
            return Compressor(**kwargs)
        return build_module(spec, **kwargs)

    def _build_indexer(self, spec: Optional[Union[ModuleSpec, type]]) -> nn.Module:
        """Build the V4 :class:`Indexer` for the CSA branch.

        See :meth:`_build_compressor` for the spec-vs-fallback contract.
        Plan-3 P21 retired the ``try/except/local Indexer`` fallback for
        the same reason.
        """
        index_topk = int(self.config.index_topk)
        index_head_dim = int(self.config.index_head_dim)
        index_n_heads = int(self.config.index_n_heads)
        kwargs = dict(
            hidden_size=self.hidden_size,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
            index_topk=index_topk,
            compress_ratio=self.compress_ratio,
        )
        if spec is None:
            return Indexer(**kwargs)
        return build_module(spec, **kwargs)

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
        """Apply partial RoPE (last ``rotary_dim`` channels) to Q and K
        using the LAYER's compress_ratio (so CSA/HCA use the compress base
        + YaRN; dense uses the main base)."""
        q = self.rope.apply_rope(q, position_ids=position_ids, compress_ratio=self.compress_ratio)
        k = self.rope.apply_rope(k, position_ids=position_ids, compress_ratio=self.compress_ratio)
        return q, k

    def _local_mask(self, S: int, *, device, dtype) -> torch.Tensor:
        """Mask for the local (SWA or full causal) branch.

        ``attn_sliding_window > 0`` enables sliding-window; ``0`` (the
        default for unit tests / configs without SWA) gives full causal.
        """
        window = self.attn_sliding_window if self.attn_sliding_window > 0 else 0
        if window > 0:
            return sliding_window_causal_mask(S, window, device=device, dtype=dtype)
        return sliding_window_causal_mask(S, S, device=device, dtype=dtype)

    def _append_sink_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Numerically-stable softmax with optional virtual-sink column.

        ``logits`` shape is ``[B, H, ..., Sk]`` — the head axis is at
        ``dim=1``. Returns probabilities on the *real* keys (sink column
        dropped) of the same shape as ``logits``.
        """
        if self.attn_sink is None:
            logits = logits - logits.amax(dim=-1, keepdim=True).detach()
            return logits.softmax(dim=-1)

        # Build a sink column that broadcasts over all dims except the
        # head axis (dim=1) and the key axis (dim=-1).
        ndim = logits.dim()
        view_shape = [1] * ndim
        view_shape[1] = self.num_heads
        view_shape[-1] = 1
        target_shape = list(logits.shape[:-1]) + [1]
        sink_col = self.attn_sink.float().view(*view_shape).expand(*target_shape)
        logits_aug = torch.cat([logits, sink_col], dim=-1)
        logits_aug = logits_aug - logits_aug.amax(dim=-1, keepdim=True).detach()
        probs = logits_aug.softmax(dim=-1)
        return probs[..., :-1]

    def _attention_forward(
        self,
        q: torch.Tensor,  # [B, H, Sq, head_dim]
        k: torch.Tensor,  # [B, H, Sk, head_dim]
        v: torch.Tensor,  # [B, H, Sk, head_dim]
        attn_mask: torch.Tensor,  # [Sq, Sk] additive (broadcasts over B,H)
    ) -> torch.Tensor:
        """Eager scaled-dot-product attention with optional attn_sink
        for the dense / HCA paths (single key axis)."""
        scale = self._attention_scale()
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        logits = logits + attn_mask
        probs = self._append_sink_softmax(logits)
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
    # compressed branches (HCA / CSA)
    # ------------------------------------------------------------------

    def _build_compressed_pool(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the compressor + compress-base partial RoPE.

        Returns ``[B, P, head_dim]`` where ``P = S // compress_ratio``.
        """
        device = hidden.device
        pooled = self.compressor(hidden)  # [B, P, head_dim]
        P = pooled.shape[1]

        # Compress-base partial RoPE on compressed indices [0..P).
        comp_pos = torch.arange(P, device=device)
        cos, sin = self.rope.compress_rope(comp_pos)
        cos = cos[..., : self.rotary_dim // 2]
        sin = sin[..., : self.rotary_dim // 2]
        pool_kv = pooled.unsqueeze(2)  # [B, P, 1, head_dim]
        pool_kv = apply_interleaved_partial_rope(pool_kv, cos, sin, rotary_dim=self.rotary_dim)
        return pool_kv.squeeze(2)  # [B, P, head_dim]

    def _hca_extra_kv(
        self,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the HCA (compress_ratio == 128) compressed branch.

        Returns ``(extra_k_bh, extra_v_bh, extra_mask)`` where the
        compressed pool is broadcast across H heads (single-latent
        compressor output) and the additive mask is shape ``[S, P]``
        (broadcasts over B, H).

        Per the techblog: pool position ``s`` covers raw tokens
        ``[s*ratio, (s+1)*ratio)``; query at raw token ``t`` may attend
        to ``s`` iff ``(s+1)*ratio - 1 <= t``.
        """
        B, S, _ = hidden.shape
        device, dtype = hidden.device, hidden.dtype
        pool = self._build_compressed_pool(hidden)  # [B, P, head_dim]
        P = pool.shape[1]

        # Broadcast pool across all H query-heads: [B, P, head_dim] -> [B, P, H, head_dim].
        pool_h = pool.unsqueeze(2).expand(B, P, self.num_heads, self.head_dim)
        # Move heads dim to dim=1: [B, H, P, head_dim].
        pool_bh = pool_h.transpose(1, 2)

        t = torch.arange(S, device=device).unsqueeze(1)  # [S, 1]
        s_end = (torch.arange(P, device=device).unsqueeze(0) + 1) * self.compress_ratio - 1  # [1, P]
        extra_mask = torch.where(s_end <= t, 0.0, float("-inf")).to(dtype)
        return pool_bh, pool_bh, extra_mask  # K = V = compressed pool

    def _csa_forward(
        self,
        hidden: torch.Tensor,
        q_bh: torch.Tensor,  # [B, H, S, head_dim]
        k_local_bh: torch.Tensor,  # [B, H, S, head_dim]
        v_local_bh: torch.Tensor,  # [B, H, S, head_dim]
        local_mask: torch.Tensor,  # [S, S]
    ) -> torch.Tensor:
        """CSA (compress_ratio == 4) joint local-SWA + sparse-compressed attention.

        The compressor produces a per-batch pool ``[B, P, head_dim]``,
        the indexer picks ``index_topk`` pool positions per query, and
        the attention runs softmax JOINTLY over ``[local_keys, sparse_keys]``
        so the optional ``attn_sink`` is shared across both branches.
        """
        B, H, S, Dh = q_bh.shape
        device, dtype = hidden.device, hidden.dtype

        # 1) Compressed pool with compress-base RoPE.
        pool = self._build_compressed_pool(hidden)  # [B, P, head_dim]
        P = pool.shape[1]

        # 2) Indexer top-K per query.
        topk_idxs, _ = self.indexer(hidden)  # [B, S, K]
        K = topk_idxs.shape[-1]
        valid = topk_idxs >= 0  # [B, S, K]
        safe_idx = topk_idxs.clamp(min=0)

        # 3) Gather per-query pool slices: [B, S, K, head_dim].
        idx_expand = safe_idx.unsqueeze(-1).expand(B, S, K, Dh)
        pool_expand = pool.unsqueeze(1).expand(B, S, P, Dh)
        gathered = torch.gather(pool_expand, dim=2, index=idx_expand)
        gathered = gathered * valid.unsqueeze(-1).to(gathered.dtype)
        sparse_mask = torch.where(valid, 0.0, float("-inf")).to(dtype)  # [B, S, K]

        scale = self._attention_scale()

        # Local logits over SWA keys: [B, H, S, S].
        local_logits = torch.matmul(q_bh.float(), k_local_bh.float().transpose(-2, -1)) * scale
        local_logits = local_logits + local_mask  # [S, S] broadcasts over B, H

        # Sparse logits over per-query top-K: [B, H, S, K].
        gathered_h = gathered.unsqueeze(1).expand(B, H, S, K, Dh).float()
        sparse_logits = torch.einsum("bhsd,bhskd->bhsk", q_bh.float(), gathered_h) * scale
        sparse_logits = sparse_logits + sparse_mask.unsqueeze(1)  # broadcast H

        # Joint softmax-with-sink across [local_keys ++ sparse_keys].
        joint_logits = torch.cat([local_logits, sparse_logits], dim=-1)  # [B, H, S, S+K]
        probs = self._append_sink_softmax(joint_logits)  # [B, H, S, S+K]

        if self.attn_dropout > 0.0 and self.training:
            probs = torch.nn.functional.dropout(probs, p=self.attn_dropout)

        probs_local = probs[..., :S].to(v_local_bh.dtype)  # [B, H, S, S]
        probs_sparse = probs[..., S:].to(v_local_bh.dtype)  # [B, H, S, K]

        # Local v-weighted sum: standard.
        out_local = torch.matmul(probs_local, v_local_bh)  # [B, H, S, head_dim]

        # Sparse v-weighted sum: per-query gather -> einsum.
        out_sparse = torch.einsum(
            "bhsk,bhskd->bhsd",
            probs_sparse,
            gathered_h.to(v_local_bh.dtype),
        )

        return out_local + out_sparse

    # ------------------------------------------------------------------
    # public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """``[B, S, D] -> [B, S, D]``.

        Dispatches on ``self.compress_ratio``:

        * ``0``   — dense / SWA over local KV (single key axis).
        * ``128`` — HCA: concat compressed pool to local KV, joint softmax.
        * ``4``   — CSA: per-query top-K from compressed pool, joint softmax.
        """
        B, S, _ = hidden.shape
        device, dtype = hidden.device, hidden.dtype

        q = self._apply_q(hidden)  # [B, S, H, head_dim]
        kv = self._apply_kv(hidden)  # [B, S, 1, head_dim]

        # Partial RoPE on Q and K. K is post-RoPE; V uses the SAME tensor
        # (V4's single-latent design: K and V share the rope-applied kv).
        q, kv = self._apply_rope_q_k(q, kv, position_ids)
        # Broadcast K / V across the H query-head axis.
        k_h = kv.expand(B, S, self.num_heads, self.head_dim)
        v_h = kv.expand(B, S, self.num_heads, self.head_dim)

        local_mask = self._local_mask(S, device=device, dtype=dtype)

        # Move heads dim before sequence: [B, S, H, head_dim] -> [B, H, S, head_dim]
        q_bh = q.transpose(1, 2)
        k_local_bh = k_h.transpose(1, 2)
        v_local_bh = v_h.transpose(1, 2)

        if self.compress_ratio == 0:
            out_bh = self._attention_forward(q_bh, k_local_bh, v_local_bh, local_mask)
        elif self.compress_ratio == 128:
            extra_k_bh, extra_v_bh, extra_mask = self._hca_extra_kv(hidden)
            k_full = torch.cat([k_local_bh, extra_k_bh], dim=2)  # along Sk
            v_full = torch.cat([v_local_bh, extra_v_bh], dim=2)
            full_mask = torch.cat([local_mask, extra_mask], dim=-1)  # [S, S+P]
            out_bh = self._attention_forward(q_bh, k_full, v_full, full_mask)
        elif self.compress_ratio == 4:
            out_bh = self._csa_forward(hidden, q_bh, k_local_bh, v_local_bh, local_mask)
        else:
            # Guarded by __init__; included for static-analysis completeness.
            raise ValueError(f"Unsupported compress_ratio {self.compress_ratio}")

        out = out_bh.transpose(1, 2).contiguous()  # [B, S, H, head_dim]
        out = out.to(dtype=dtype)

        if self.linear_o_a is not None:
            return self._grouped_o_projection(out)
        return self._flat_o_projection(out)


__all__ = [
    "DeepseekV4Attention",
    "DeepseekV4AttentionSubmodules",
]
