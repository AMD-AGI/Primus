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

import atexit
import collections
import logging
import math
import os
import statistics
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from megatron.core.transformer.enums import AttnMaskType
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
from primus.backends.megatron.core.transformer.indexer_distill_loss import (
    V4IndexerLossAutoScaler,
    compute_indexer_distill_loss,
    log_indexer_distill_loss,
)
from primus.backends.megatron.core.transformer.keep_in_fp32 import (
    KeepInFp32Mixin,
    mark_keep_in_fp32,
    unmark_keep_in_fp32,
)
from primus.backends.megatron.core.transformer.local_rmsnorm import LocalRMSNorm
from primus.backends.megatron.core.transformer.sliding_window_kv import (
    sliding_window_causal_mask,
)

# All attention backend entries come from the kernels package __init__ (the
# single entry point): eager, triton v1/v2. Naming: v4_attention_<be>
# (dense/HCA) and v4_csa_attention_<be> (CSA). The gluon backend is NOT imported
# here — it hard-depends on triton.experimental.gluon (gfx950 only) and is loaded
# lazily via load_gluon_attention_backends() only when a layer selects it.
from primus.backends.megatron.core.transformer.v4_attention_kernels import (
    eager_v4_attention,
    eager_v4_csa_attention,
    load_flydsl_attention_backends,
    load_gluon_attention_backends,
    load_gluon_v2_attention_backends,
    load_gluon_v3_attention_backends,
    load_turbo_attention_backends,
    v4_attention_v1,
    v4_attention_v2,
    v4_csa_attention_v0,
    v4_csa_attention_v1,
    v4_csa_attention_v2,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.rmsnorm import (
    fused_rms_norm,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.rope_interleaved_partial import (
    apply_rope_from_positions,
)

_SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)

logger = logging.getLogger(__name__)


def _v4_get_cp_group():
    from primus.backends.megatron.core.transformer.deepseek_v4_cp import get_cp_group

    return get_cp_group()


def _v4_exchange_boundary_kv(kv, d_window, cp_group):
    from primus.backends.megatron.core.transformer.deepseek_v4_cp import (
        exchange_boundary_kv,
    )

    return exchange_boundary_kv(kv, d_window, cp_group)


def _require_gfx950() -> None:
    """Assert the current device is gfx950 / CDNA4 before using the gluon backend.

    The gluon sparse-MLA kernels are hand-tuned for gfx950 (MI350/MI355X); running
    them on any other arch is unsupported. Called only when a layer selects
    ``use_v4_attention_backend`` / ``use_v4_csa_attention_backend = 'gluon'``.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "use_v4_attention_backend / use_v4_csa_attention_backend = 'gluon' requires a "
            "CUDA/HIP gfx950 (CDNA4) device, but no accelerator is available. Select "
            "eager | triton_v1 | triton_v2 instead."
        )
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    if "gfx950" not in arch:
        raise RuntimeError(
            "use_v4_attention_backend / use_v4_csa_attention_backend = 'gluon' targets gfx950 / "
            f"CDNA4 (MI350/MI355X); got device arch {arch!r}. Select eager | triton_v1 | triton_v2 "
            "instead, or run on gfx950."
        )


# ---------------------------------------------------------------------------
# P32 diagnostic: collect in-context cuda.Event timings of v4_attention_v1
# ---------------------------------------------------------------------------


class _DeepseekV4AttentionDiag:
    """Accumulator for ``PRIMUS_V4_DIAG_TIME=1`` per-call timings."""

    _per_mode: dict[str, list[float]] = collections.defaultdict(list)
    _registered: bool = False
    shape_logged: dict[str, bool] = {}

    @classmethod
    def record(cls, *, mode: str, ms: float, swa: int) -> None:
        cls._per_mode[mode].append(ms)
        if not cls._registered:
            cls._registered = True
            atexit.register(cls.dump)

    @classmethod
    def dump(cls) -> None:
        if not cls._per_mode:
            return
        rank = os.environ.get("RANK", "0")
        try:
            local_rank = int(rank)
        except (TypeError, ValueError):
            local_rank = 0
        if local_rank != 0:
            return
        print("[PRIMUS_V4_DIAG_TIME] v4_attention_v1 inline cuda.Event timings:", flush=True)
        for mode, vs in cls._per_mode.items():
            if not vs:
                continue
            # Drop first 3 to skip warmup.
            stable = vs[3:] if len(vs) > 3 else vs
            print(
                f"  mode={mode:<6s}  n={len(vs):4d}  "
                f"all_med={statistics.median(vs):7.3f}ms  "
                f"warm_med={statistics.median(stable):7.3f}ms  "
                f"warm_min={min(stable):7.3f}ms  "
                f"warm_max={max(stable):7.3f}ms",
                flush=True,
            )


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

    Plan-3 P22: ``core_attention`` is the Turbo / TE flash-attention
    kernel.  Only the dense layer kind (``compress_ratio == 0``) emits
    a spec for this slot — HCA / CSA cannot use a stock flash-attn
    kernel (HCA needs a joint sink across two key streams which would
    require an LSE-returning flash kernel; CSA needs per-query top-K
    indexed keys which is not a flash pattern).  When the dense path
    receives a ``core_attention`` it bypasses the eager-Python softmax
    and runs through ``provider.core_attention()`` instead.  When
    ``provider.core_attention()`` returns
    :class:`PrimusTurboAttention` (i.e. ``use_turbo_attention=True``)
    and V4's ``attn_sink`` is on, the attention module aliases
    ``core_attention.sinks`` to ``self.attn_sink`` so the released
    checkpoint key path is preserved.
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
    # Plan-3 P22: dense (compress_ratio == 0) layers only.
    core_attention: Optional[Union[ModuleSpec, type]] = None


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


def _v4_o_a_fp8_enabled(config) -> bool:
    """Whether to run the grouped-O ``o_a`` down-projection in MXFP8.

    ``o_a`` is a per-group (batched) matmul done as a manual einsum on
    ``linear_o_a.weight``, so it bypasses the fp8 linear path and stays bf16.
    When PRIMUS_V4_FP8_ATTN_PROJ is set (and TP=1, where the surrounding
    projections are already routed to fp8) and the layer is in turbo-fp8, run
    it as per-group fp8 GEMMs instead. Default off.
    """
    if os.environ.get("PRIMUS_V4_FP8_ATTN_PROJ", "0") != "1":
        return False
    if getattr(config, "tensor_model_parallel_size", 1) != 1:
        return False
    try:
        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboLowPrecisionGlobalStateManager as _M,
        )

        return _M.is_turbo_fp8_enabled()
    except Exception:
        return False


def _fp8_grouped_o_a(attn_g: torch.Tensor, wo_a_w: torch.Tensor) -> torch.Tensor:
    """Fused MXFP8 grouped-O ``o_a`` down-projection (replaces the bf16 einsum).

    ``attn_g`` [B,S,G,d], ``wo_a_w`` [G,r,d] -> [B,S,G,r]. The G groups are an
    independent batched matmul ``[B*S,d] @ [d,r]`` per group; we run them as a
    SINGLE ``grouped_gemm_fp8`` (one fused Triton launch) instead of a per-group
    ``gemm_fp8`` loop (G launches + G× quant). Stack tokens group-major into
    ``[G*B*S, d]`` with ``group_lens=[B*S]*G``; weight ``[G,d,r]`` (trans_b=False).
    d=(H*head_dim)/G and B*S are multiples of 32, so the MX block scale is clean.
    """
    import primus_turbo.pytorch as pt

    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboLowPrecisionGlobalStateManager as _M,
    )

    cfg = _M.get_turbo_quant_config().data()
    B, S, G, d = attn_g.shape
    r = wo_a_w.shape[1]
    a = attn_g.permute(2, 0, 1, 3).reshape(G * B * S, d).contiguous()  # [G*BS, d], group-major
    b = wo_a_w.transpose(1, 2).contiguous()  # [G, d, r]  (K=d, N=r, trans_b=False)
    group_lens = torch.full((G,), B * S, dtype=torch.int64, device=a.device)
    out = pt.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=False, config=cfg)  # [G*BS, r]
    return out.reshape(G, B, S, r).permute(1, 2, 0, 3)  # [B, S, G, r]


def _coerce_optional_bool_flag(value: object, *, field_name: str) -> bool:
    """Coerce a possibly-stringified yaml flag to a clean ``bool``.

    Yaml interpolation like ``${PRIMUS_FOO:false}`` resolves to the
    STRING ``"false"`` when the env var is unset, and the naive
    ``bool("false") is True`` would silently flip a default-off knob
    to on.  Accept the common string spellings explicitly and treat
    everything else as truthy/falsy via the normal ``bool(...)`` rule.

    Plan-8 P57 close-out 2 added this helper for the new
    ``use_v4_tilelang_*`` flags; existing flags
    (``use_v4_triton_*`` / ``use_v4_compiled_sinkhorn``) avoid the
    issue because the V4 run scripts always pass them via
    ``--<flag> "False"`` and the override parser coerces to a Python
    ``False`` before the config ever sees a string.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("0", "false", "no", "off", ""):
            return False
        if lowered in ("1", "true", "yes", "on"):
            return True
        raise ValueError(
            f"Unrecognised string value for boolean config flag "
            f"{field_name!r}: {value!r}; expected one of "
            "'true' / 'false' / '1' / '0' / 'yes' / 'no' / 'on' / 'off'."
        )
    return bool(value)


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

    Small-kernel-fusion (2026-07-03): the eager chain (bf16->fp32 cast +
    square + mean + rsqrt + mul + fp32->bf16 cast, ~6 kernels / call ×
    8 attention layers) is collapsed into one Triton FWD + one BWD kernel
    via :func:`fused_rms_norm` (parameter-less, ``out_dtype = in_dtype``).
    Gated by ``PRIMUS_RMSNORM_TRITON`` (default on); the dispatcher falls
    back to the bit-identical eager body on CPU / when the knob is off.
    """
    return fused_rms_norm(x, None, eps=eps, mid_cast=False, out_dtype=x.dtype)


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


class DeepseekV4Attention(KeepInFp32Mixin, MLASelfAttention):
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

    **Plan-4 P27 — kernel dispatch precedence.**

    The softmax-and-attend kernel each layer fires through is selected
    in :meth:`forward` / :meth:`_csa_forward` based on three independent
    config flags (``use_turbo_attention``, ``use_v4_triton_attention``,
    ``use_v4_triton_csa_attention``).  The layer-kind-specific
    precedence is:

    .. code-block:: text

        compress_ratio == 0  (dense / SWA, single key axis):
            use_turbo_attention      > use_v4_triton_attention > eager
            (-> self.core_attention)   (-> v4_attention_v1)         (-> _attention_forward)

        compress_ratio == 128  (HCA: local SWA + full compressed pool):
            use_v4_triton_attention  > eager
            (-> v4_attention_v1,          (-> _attention_forward
             HCA path with joint        with cat([local, pool])
             [local | pool] mask)       additive mask)

            ``use_turbo_attention`` does NOT route HCA — Turbo's
            flash-attn returns no LSE so the joint local+pool softmax
            cannot be decomposed into two flash calls.

        compress_ratio == 4  (CSA: local SWA + per-query top-K gather):
            use_v4_triton_csa_attention  > eager
            (-> v4_csa_attention_v0)          (-> _csa_forward eager)

            Neither ``use_turbo_attention`` nor
            ``use_v4_triton_attention`` applies to CSA — the per-query
            top-K gather (``gathered = pool[..., topk_idxs, :]``) is
            sparse-per-row indexed attention with no flash-attn
            equivalent.

    Auto-disable rules (init-side, fail-loud):

    * ``use_v4_triton_attention=True`` + ``compress_ratio == 4`` →
      auto-disabled (CSA layers must opt in via the separate flag).
    * ``use_v4_triton_csa_attention=True`` + ``compress_ratio != 4`` →
      auto-disabled (the dense / HCA flag is ``use_v4_triton_attention``).

    On rank 0 each ``__init__`` emits one ``INFO`` log line through
    :meth:`_log_kernel_choice` summarising the dispatch outcome for
    the layer (e.g. ``[V4-attn] Layer 17: cr=128, kernel = v4_attention_v1
    (Triton, HCA path)``) so smoke / training logs unambiguously show
    which kernel each layer is firing through.
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
        # ---- P14: head sharding across TP ----------------------------------
        # Default (v4_shard_attention_heads off): every rank materialises all
        # `num_heads` heads, because linear_q_up_proj gathers its output back to full
        # width. TP then shards weights only, and the [B, S, H, head_dim] query is
        # replicated -- 64 KiB/token at V4-Flash width, which is what caps the usable
        # sequence length. With P14 on, each rank owns num_heads/tp heads end to end.
        from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_layer_specs import (
            v4_shard_heads as _v4_shard_heads,
        )

        self.shard_heads = _v4_shard_heads(config)
        self.tp_size = int(getattr(config, "tensor_model_parallel_size", 1) or 1) if self.shard_heads else 1
        num_heads_local = num_heads // self.tp_size

        self.hidden_size = hidden_size
        self.num_heads = num_heads_local
        self.num_heads_global = num_heads
        self.num_attention_heads_per_partition = num_heads_local
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
        #
        # FP32 in the released checkpoint, and the gluon / flydsl_v1 kernels
        # assert ``attn_sink.dtype == torch.float32`` at their entry. That
        # assertion is satisfied by the promotion the callers already do
        # (``sink.float()`` in the sparse-MLA adapter and the eager reference),
        # so the parameter itself may follow the model dtype. Pinning it to FP32
        # is therefore opt-in via ``PRIMUS_V4_KEEP_FP32`` -- see
        # ``keep_in_fp32`` for why holding a second parameter dtype is not free.
        if attn_sink_enabled:
            # One sink per head, so it shards with the heads under P14 -- hence
            # num_heads_local, not num_heads. Kept in fp32 (and marked so the fp16
            # module wrapper leaves it alone) as upstream does.
            self.attn_sink = nn.Parameter(torch.zeros(num_heads_local, dtype=torch.float32))
            mark_keep_in_fp32(self.attn_sink)
        else:
            self.register_parameter("attn_sink", None)

        # ---- compressor / indexer (compressed branches only) ----
        self.compressor: Optional[nn.Module] = None
        self.indexer: Optional[nn.Module] = None
        # Last indexer distillation loss (detached), for logging. None until a
        # training step runs with the loss enabled.
        self.last_indexer_distill_loss: Optional[torch.Tensor] = None
        if self.compress_ratio > 0:
            self.compressor = self._build_compressor(submodules.compressor)
            if self.compress_ratio == 4:
                self.indexer = self._build_indexer(submodules.indexer)
                # ``topk`` is not differentiable and the forward only consumes
                # ``topk_idxs``, so the Indexer can only learn through the
                # distillation loss (see ``indexer_distill_loss``). Without it,
                # leaving the params trainable inserts permanently-dead params
                # into the distributed-optimizer grad buckets: that wastes grad /
                # optimizer state plus cross-node grad-sync bandwidth AND trips
                # Megatron's overlap_grad_reduce invariant (every bucket param
                # must fire its grad-ready backward hook), which is what forced
                # overlap_grad_reduce / param_gather OFF and crippled cross-node
                # DP scaling. So freeze unless the loss is actually enabled.
                # PRIMUS_V4_INDEXER_TRAINABLE=1 still forces trainable.
                #
                # PRIMUS_V4_DISTILL_FREEZE_INDEXER=1 additionally freezes the
                # indexer *while still computing the loss*. That is not a
                # training mode -- the indexer then learns nothing -- it exists
                # to measure how much of the loss's cost is its backward, by
                # A/B-ing against the same run without it.
                _diag_freeze = os.environ.get("PRIMUS_V4_DISTILL_FREEZE_INDEXER", "0") == "1"
                if _diag_freeze or (
                    not self.indexer_distill_enabled
                    and os.environ.get("PRIMUS_V4_INDEXER_TRAINABLE", "0") != "1"
                ):
                    for _indexer_param in self.indexer.parameters():
                        _indexer_param.requires_grad_(False)

                # Most of what training the indexer costs is the autograd engine
                # walking the ~20 nodes its forward builds, per CSA layer, not
                # arithmetic; torch.compile lets AOTAutograd emit one fused
                # backward instead. `topk` is data-dependent and graph-breaks,
                # which is fine -- everything before it is what costs.
                if self.indexer_distill_enabled and os.environ.get("PRIMUS_V4_INDEXER_COMPILE", "0") == "1":
                    self.indexer = torch.compile(self.indexer, dynamic=False)

        # ---- core attention (Turbo / TE flash) — dense layers only ----
        # Plan-3 P22: when the spec emits a ``core_attention`` submodule
        # (only on dense ``compress_ratio == 0`` layers), build it now and
        # use it as the softmax-and-attend kernel instead of the
        # eager-Python ``_attention_forward``.  HCA + CSA always run
        # eager-Python because their joint softmax / per-query top-K
        # gather can't be expressed as a stock flash-attn call (see
        # comments in ``forward`` / ``_csa_forward``).
        #
        # The constant ``softmax_scale`` is precomputed via
        # ``_attention_scale()`` (the YaRN ``m_scale`` is a layer-static
        # constant set at RoPE init time, so this matches the eager-path
        # scale exactly for ``compress_ratio == 0``).
        # Plan-4 P25: in-tree Primus Triton kernel for cr ∈ {0, 128}.
        # Read the config flag once at __init__ so ``forward`` only does
        # a cheap attribute load. Precedence in ``forward`` is
        # ``use_turbo_attention > use_v4_triton_attention > eager``.
        # ---- attention backend selection (unified string selectors) ----
        # ``use_v4_attention_backend`` selects the dense (cr=0) / HCA (cr=128)
        # kernel; ``use_v4_csa_attention_backend`` selects the CSA (cr=4) kernel.
        # ``use_turbo_attention`` (built as ``core_attention`` below) still takes
        # precedence for the dense path when it can be built.
        _ATTN_BACKENDS = (
            "eager",
            "triton_v1",
            "triton_v2",
            "gluon",
            "gluon_v2",
            "gluon_v3",
            "flydsl_v1",
            "turbo",
        )
        _CSA_BACKENDS = (
            "eager",
            "triton_v0",
            "triton_v1",
            "triton_v2",
            "gluon",
            "gluon_v2",
            "gluon_v3",
            "flydsl_v0",
            "flydsl_v1",
            "turbo",
        )
        self._attn_backend: str = str(getattr(config, "use_v4_attention_backend", "triton_v1") or "triton_v1")
        self._csa_backend: str = str(
            getattr(config, "use_v4_csa_attention_backend", "triton_v1") or "triton_v1"
        )
        if self._attn_backend not in _ATTN_BACKENDS:
            raise ValueError(
                f"use_v4_attention_backend={self._attn_backend!r} is not a valid dense/HCA backend; "
                f"expected one of {_ATTN_BACKENDS}"
            )
        if self._csa_backend not in _CSA_BACKENDS:
            raise ValueError(
                f"use_v4_csa_attention_backend={self._csa_backend!r} is not a valid CSA backend; "
                f"expected one of {_CSA_BACKENDS}"
            )

        # gluon is a gfx950/CDNA4-only backend with a hard triton.experimental.gluon
        # dependency. Load it (and validate the arch) ONLY when a layer actually
        # selects it, so non-gluon backends never pay the import and never crash on
        # unsupported hardware / Triton builds. The loader raises a clear ImportError
        # if the dependency is missing; ``_require_gfx950`` raises if the arch is wrong.
        self._v4_attention_gluon = None
        self._v4_csa_attention_gluon = None
        if "gluon" in (self._attn_backend, self._csa_backend):
            _require_gfx950()
            self._v4_attention_gluon, self._v4_csa_attention_gluon = load_gluon_attention_backends()

        # gluon_v2 (2nd-gen gluon fwd+bwd) — same gfx950-only lazy-load contract as gluon.
        self._v4_attention_gluon_v2 = None
        self._v4_csa_attention_gluon_v2 = None
        if "gluon_v2" in (self._attn_backend, self._csa_backend):
            _require_gfx950()
            self._v4_attention_gluon_v2, self._v4_csa_attention_gluon_v2 = load_gluon_v2_attention_backends()

        # gluon_v3 (3rd-gen optimized gluon fwd+bwd) — same gfx950-only lazy-load contract.
        self._v4_attention_gluon_v3 = None
        self._v4_csa_attention_gluon_v3 = None
        if "gluon_v3" in (self._attn_backend, self._csa_backend):
            _require_gfx950()
            self._v4_attention_gluon_v3, self._v4_csa_attention_gluon_v3 = load_gluon_v3_attention_backends()

        # flydsl_v1 (native FlyDSL MFMA) is likewise a gfx950/CDNA4-only backend with
        # a hard `flydsl` pip dependency; load + arch-validate only when selected.
        self._v4_attention_flydsl = None
        self._v4_csa_attention_flydsl = None
        if "flydsl_v1" in (self._attn_backend, self._csa_backend):
            _require_gfx950()
            self._v4_attention_flydsl, self._v4_csa_attention_flydsl = load_flydsl_attention_backends()

        # turbo (Primus-Turbo native-FlyDSL sparse-MLA via the turbo API) — same gfx950-only
        # lazy-load contract; hard-depends on the installed primus_turbo (flydsl attention) + flydsl.
        self._v4_attention_turbo = None
        self._v4_csa_attention_turbo = None
        if "turbo" in (self._attn_backend, self._csa_backend):
            _require_gfx950()
            self._v4_attention_turbo, self._v4_csa_attention_turbo = load_turbo_attention_backends()

        self.core_attention: Optional[nn.Module] = None
        self._use_core_attention: bool = False
        if submodules.core_attention is not None and self.compress_ratio == 0:
            softmax_scale = self._attention_scale()
            try:
                self.core_attention = build_module(
                    submodules.core_attention,
                    config=config,
                    layer_number=self.layer_number,
                    attn_mask_type=AttnMaskType.causal,
                    attention_type="self",
                    softmax_scale=softmax_scale,
                    k_channels=head_dim,
                    v_channels=head_dim,
                    cp_comm_type="p2p",
                    pg_collection=self.pg_collection,
                )
            except TypeError:
                # Some core-attention classes (e.g. local CPU stubs in
                # unit tests) don't accept the full TE / Turbo kwarg set.
                # Retry with the minimal kwargs Megatron ships everywhere.
                self.core_attention = build_module(
                    submodules.core_attention,
                    config=config,
                    layer_number=self.layer_number,
                    attn_mask_type=AttnMaskType.causal,
                    attention_type="self",
                    softmax_scale=softmax_scale,
                )

            # Sink alias: when V4's per-head learnable sink is on AND the
            # core-attention class supports learned sinks (Turbo only —
            # the TE class does not), tie ``core_attention.sinks`` to
            # ``self.attn_sink`` so the released-checkpoint key
            # ``layers.{i}.attn.attn_sink`` keeps loading.  TE classes
            # that don't expose ``use_sink_attention`` get ``False`` here
            # and we fall back to eager-Python so the inline
            # softmax-with-sink path still produces the right math.
            core_use_sink = bool(getattr(self.core_attention, "use_sink_attention", False))
            if attn_sink_enabled and core_use_sink:
                # Aliasing means Turbo and V4 share one Parameter object, so the
                # sink has to carry the dtype Turbo allocated for its own
                # ``sinks``. That is the one configuration where the sink cannot
                # stay FP32, so drop the keep-in-FP32 mark as well -- otherwise
                # the model-wide dtype conversion would silently undo the cast
                # and hand Turbo a tensor it did not allocate for. The eager
                # path promotes to float32 inside the softmax either way, so
                # only the stored resolution differs.
                turbo_sinks = getattr(self.core_attention, "sinks", None)
                if turbo_sinks is not None and turbo_sinks.dtype != self.attn_sink.dtype:
                    unmark_keep_in_fp32(self.attn_sink)
                    self.attn_sink.data = self.attn_sink.data.to(turbo_sinks.dtype)
                self.core_attention.sinks = self.attn_sink
                self._use_core_attention = True
            elif not attn_sink_enabled and self.core_attention is not None:
                # No-sink V4 still uses core_attention (e.g. unit tests,
                # ablations).  SWA is honored by Turbo only when sinks are
                # on, so we accept this only when SWA is off too.
                if self.attn_sliding_window <= 0:
                    self._use_core_attention = True

        # Plan-4 P27: surface the dispatch outcome in the training log
        # so smoke / debug logs unambiguously show which kernel each
        # layer is firing through (precedence is documented in the
        # class docstring).  Rank-0 only — every rank's own
        # per-rank-file already captures the right entries.
        self._log_kernel_choice()

    # ------------------------------------------------------------------
    # construction helpers (compressed branches)
    # ------------------------------------------------------------------

    def _log_kernel_choice(self) -> None:
        """Emit one ``INFO`` log line summarising this layer's kernel choice.

        Plan-4 P27.  Resolves the precedence outcome captured by
        :meth:`forward` / :meth:`_csa_forward` (see class docstring) and
        logs it once at ``__init__`` time so smoke / training logs
        unambiguously show which kernel is firing for each layer.

        Rank-0 only when distributed; in single-process unit tests the
        log fires unconditionally so ``caplog.at_level(logging.INFO)``
        captures it.  We cannot use Megatron's ``print_rank_0`` here
        because this module is also imported in CPU-only unit tests
        where Megatron's parallel-state isn't initialised.
        """
        try:
            dist_initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
        except Exception:
            dist_initialized = False
        if dist_initialized and torch.distributed.get_rank() != 0:
            return

        if self.compress_ratio == 0:
            if self._use_core_attention:
                kernel = "core_attention (Turbo / TE flash)"
            else:
                kernel = f"dense attention backend = {self._attn_backend}"
        elif self.compress_ratio == 128:
            kernel = f"HCA attention backend = {self._attn_backend}"
        elif self.compress_ratio == 4:
            kernel = f"CSA attention backend = {self._csa_backend}"
        else:
            # Defensive: __init__ already raises ValueError for unsupported
            # compress_ratio, so this branch should be unreachable.
            kernel = f"<unknown compress_ratio={self.compress_ratio}>"

        logger.info(
            "[V4-attn] Layer %s: cr=%s, kernel = %s",
            self.layer_number,
            self.compress_ratio,
            kernel,
        )

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
            use_fp8_qk=bool(getattr(self.config, "use_v4_fp8_indexer", False)),
            # P14: hand the indexer the TP group so it can shard its heads and
            # all-reduce the partial score sums. None (the default) keeps the
            # replicated, unsharded behaviour.
            tp_group=self._v4_tp_group() if self.shard_heads else None,
            # The indexer rotates its own Q / K with this layer's compressed
            # RoPE before scoring, so it needs the same cache the compressed
            # pool uses. Passed by reference; the Indexer does not register it.
            rope=self.rope.get_rope(compress_ratio=self.compress_ratio),
            rotary_dim=self.rotary_dim,
        )
        if spec is None:
            return Indexer(**kwargs)
        return build_module(spec, **kwargs)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _v4_tp_group():
        """Tensor-parallel process group, or None when TP is off / dist is not up."""
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return None
        from megatron.core import parallel_state

        try:
            g = parallel_state.get_tensor_model_parallel_group()
        except (AssertionError, RuntimeError):
            return None
        return g if g is not None and g.size() > 1 else None

    @property
    def rope(self) -> DualRoPE:
        return self._rope[0]

    @property
    def indexer_distill_coeff(self) -> float:
        """Coefficient of the indexer distillation loss (``0`` = disabled)."""
        return float(getattr(self.config, "v4_indexer_distill_loss_coeff", 0.0) or 0.0)

    @property
    def indexer_distill_enabled(self) -> bool:
        """True when this layer trains its indexer via the distillation loss."""
        return self.compress_ratio == 4 and self.indexer_distill_coeff > 0.0

    def _indexer_loss_head_group(self):
        """Group the attention heads are sharded over, or ``None`` if they are not.

        ``linear_q_up_proj`` is column-parallel with ``gather_output=True``, so
        every TP rank holds all heads and the distillation loss's head sum is
        already complete -- unlike the open-source reference, whose attention
        keeps them sharded and therefore all-reduces. Resolve the group anyway
        when the shapes say the heads *are* sharded, so dropping the gather
        cannot silently turn the target distribution into a partial sum.
        """
        if self.num_attention_heads_per_partition == int(self.config.num_attention_heads):
            return None
        group = getattr(self.pg_collection, "tp", None)
        if group is not None:
            return group
        from megatron.core import parallel_state

        return parallel_state.get_tensor_model_parallel_group()

    def _log_indexer_distill_loss(self, device: torch.device) -> None:
        """Report this layer's indexer loss, zero on layers without an indexer.

        Every V4 layer reports so the tracker key exists on every pipeline rank
        -- see :func:`log_indexer_distill_loss`.
        """
        if self.indexer_distill_coeff <= 0.0:
            return
        if not (self.training and torch.is_grad_enabled()):
            # Matches the condition the loss is computed under, so a stale
            # value from an earlier step can never be reported.
            return
        mtp_layers = getattr(self.config, "mtp_num_layers", 0) or 0
        log_indexer_distill_loss(
            self.last_indexer_distill_loss,
            layer_number=self.layer_number,
            num_layers=int(self.config.num_layers) + int(mtp_layers),
            device=device,
        )

    def _attention_scale(self) -> float:
        """Softmax temperature for every branch: plain ``1 / sqrt(head_dim)``.

        V4 keeps attention logits in range via the Q / KV RMSNorms, so the
        YaRN magnitude factor (``m_scale``) is deliberately NOT folded in here
        -- ``inference/model.py`` uses ``self.softmax_scale = head_dim ** -0.5``
        for both the core attention and the indexer. The YaRN *frequency*
        interpolation on ``inv_freq`` is unaffected.
        """
        return 1.0 / math.sqrt(self.head_dim)

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
        for the dense / HCA paths (single key axis).

        Plan-4 P24: math lives in
        :func:`primus...v4_attention_kernels._eager.reference.eager_v4_attention`
        so the dense / HCA path, the plan-4 Triton kernel (P25), and the
        plan-4 unit-test harness share one definition. The caller has
        already pre-built the ``[Sq, Sk]`` additive mask (SWA-causal
        for dense, ``cat([local_mask, hca_mask])`` for HCA) so we pass
        ``swa_window=0`` and let the reference op use the supplied
        ``additive_mask`` directly — bit-identical to the pre-P24
        inline implementation.
        """
        return eager_v4_attention(
            q,
            k,
            v,
            sink=self.attn_sink,
            swa_window=0,
            additive_mask=attn_mask,
            attn_dropout=self.attn_dropout,
            training=self.training,
            scale=self._attention_scale(),
        )

    def _attention_forward_via_v4_triton(
        self,
        q: torch.Tensor,  # [B, H, Sq, head_dim]
        k: torch.Tensor,  # [B, H, Sk, head_dim]
        v: torch.Tensor,  # [B, H, Sk, head_dim]
        attn_mask: Optional[torch.Tensor],  # [Sq, Sk] additive (broadcasts over B, H)
        *,
        swa_window: int = 0,
        hca_local_seqlen: int = 0,
    ) -> torch.Tensor:
        """Run the dense / HCA softmax-and-attend through the plan-4
        in-tree :func:`v4_attention_v1` Triton kernel.

        Numerically equivalent to :meth:`_attention_forward` (same eager
        ``q @ k^T * scale + mask + sink → softmax → @ v`` math) but
        executes in a single fused kernel that re-materialises ``P``
        from the saved LSE during the BWD instead of storing the
        ``[Sq, Sk]`` ``P`` tensor — important at full V4-Flash dims
        (``S=4096`` ⇒ ``P`` is 32 MiB / microbatch).

        Plan-5 P30 flips the dense path to ``attn_mask=None`` +
        ``swa_window > 0`` so the kernel can skip K tiles that are
        guaranteed outside the sliding window. HCA uses the same pruning
        for its local prefix by passing a pool-only mask plus
        ``hca_local_seqlen``; the kernel then runs local SWA and pool
        visibility as two loops under one joint softmax.
        """
        # Plan-5 P32: opt-in microbench-vs-proxy timing harness, gated
        # by ``PRIMUS_V4_DIAG_TIME=1``. Adds a synchronous cuda.Event
        # span around the kernel call and dumps per-mode median/min/max
        # at process exit (rank 0 only). Used to root-cause the dual-RoPE
        # bf16 -> fp32 upcast bug that made every V4 attention kernel
        # run 1.8-7x slower in the proxy than in the standalone bench;
        # left in-tree for future microbench-vs-proxy regressions.
        if os.environ.get("PRIMUS_V4_DIAG_TIME", "0") == "1":
            mode = "hca" if hca_local_seqlen > 0 else "dense"
            if not _DeepseekV4AttentionDiag.shape_logged.get(mode, False):
                _DeepseekV4AttentionDiag.shape_logged[mode] = True
                print(
                    f"[PRIMUS_V4_DIAG_TIME] mode={mode}  "
                    f"q={tuple(q.shape)}/{q.dtype}/contig={q.is_contiguous()}  "
                    f"k={tuple(k.shape)}/{k.dtype}/contig={k.is_contiguous()}  "
                    f"v={tuple(v.shape)}/{v.dtype}/contig={v.is_contiguous()}  "
                    f"swa={swa_window} hca_local={hca_local_seqlen}",
                    flush=True,
                )
            torch.cuda.synchronize()
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            out = v4_attention_v1(
                q,
                k,
                v,
                sink=self.attn_sink,
                swa_window=int(swa_window) if (attn_mask is None or hca_local_seqlen > 0) else 0,
                additive_mask=attn_mask,
                attn_dropout=self.attn_dropout,
                training=self.training,
                scale=self._attention_scale(),
                hca_local_seqlen=int(hca_local_seqlen),
            )
            ev_e.record()
            torch.cuda.synchronize()
            _DeepseekV4AttentionDiag.record(mode=mode, ms=ev_s.elapsed_time(ev_e), swa=swa_window)
            return out
        return v4_attention_v1(
            q,
            k,
            v,
            sink=self.attn_sink,
            swa_window=int(swa_window) if (attn_mask is None or hca_local_seqlen > 0) else 0,
            additive_mask=attn_mask,
            attn_dropout=self.attn_dropout,
            training=self.training,
            scale=self._attention_scale(),
            hca_local_seqlen=int(hca_local_seqlen),
        )

    def _attention_forward_via_core(
        self,
        q: torch.Tensor,  # [B, S, H, head_dim] (post-RoPE)
        kv: torch.Tensor,  # [B, S, 1, head_dim] (post-RoPE, single-latent)
    ) -> torch.Tensor:
        """Run the dense (compress_ratio == 0) softmax-and-attend through
        ``self.core_attention`` (Turbo flash-attn / TE flash-attn).

        Plan-3 P22.  Avoids materialising the eager
        ``[B, H, S, S] fp32`` logits tensor — at full V4-Flash dims
        (``H=64, S=4096, hc_mult=4``) that's 16 GiB / microbatch, and
        the dominant activation cost.

        Inputs use V4's local-frame layout (Q has all H heads, KV is
        single-latent with 1 head).  We forward as Turbo's required
        ``qkv_format="sbhd"`` and let the underlying flash kernel
        broadcast the 1-head KV across H query heads (MQA).  Causal
        masking + (optional) sliding window are honored by the kernel
        directly — the eager ``local_mask`` is not used here.

        Returns ``[B, H, S, head_dim]`` to match the contract of
        :meth:`_attention_forward`.
        """
        B, S, H, Dh = q.shape
        # [B, S, H, D] -> [S, B, H, D] (qkv_format="sbhd").
        q_sbhd = q.transpose(0, 1).contiguous()
        kv_sbhd = kv.transpose(0, 1).contiguous()  # [S, B, 1, D]

        # Turbo / TE flash-attn forward.  ``attention_mask=None`` is
        # legal for causal+SWA (the kernel builds the mask internally
        # from ``attn_mask_type`` + the layer's ``window_size``).
        out = self.core_attention(
            q_sbhd,
            kv_sbhd,
            kv_sbhd,
            None,
            attn_mask_type=AttnMaskType.causal,
        )  # -> [S, B, H * head_dim]

        # [S, B, H*D] -> [B, S, H, D] -> [B, H, S, D].
        out = out.view(S, B, H, Dh).permute(1, 2, 0, 3).contiguous()
        return out

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
        # Under P14 this rank owns o_groups/tp groups and H is already the local head
        # count, so H*Dh/G_local is the same n_per_group as the unsharded path -- the
        # group WIDTH is a property of o_groups, not of TP.
        G = self.o_groups // self.tp_size
        attn_g = attn.reshape(B, S, G, (H * Dh) // G)  # [B, S, G_local, H*Dh/G_local]

        wo_a = self.linear_o_a
        weight = wo_a.weight if hasattr(wo_a, "weight") else None
        if weight is None:
            # Fall back to a dense linear apply (Megatron parallel linears
            # without a directly accessible weight attribute).
            o = _projection_forward(wo_a, attn_g.reshape(B, S, -1))
            o = o.view(B, S, G * self.o_lora_rank)
        else:
            wo_a_w = weight.view(G, self.o_lora_rank, (H * Dh) // G)  # G is local
            if _v4_o_a_fp8_enabled(self.config):
                o = _fp8_grouped_o_a(attn_g, wo_a_w)  # per-group MXFP8
            else:
                o = torch.einsum("bsgd,grd->bsgr", attn_g, wo_a_w)
            o = o.flatten(2)
        return _projection_forward(self.linear_o_b, o)

    def _flat_o_projection(self, attn: torch.Tensor) -> torch.Tensor:
        """MLA-style flat output projection (``o_lora_rank == 0`` fast path)."""
        B, S, H, Dh = attn.shape
        return _projection_forward(self.linear_proj, attn.reshape(B, S, H * Dh))

    def _apply_inverse_rope(self, attn: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """De-rotate the core-attention output (RoPE at ``position = -t``).

        V4 shares a single latent between K and V, so the values entering the
        softmax are already rotated and the naive output

            ``o_t = sum_s P[t,s] * R(s) v_s``

        carries *absolute* positions. Rotating it by ``-t`` turns every
        contribution into ``R(s - t) v_s``, i.e. the relative encoding the
        model expects. The released ``inference/model.py`` does exactly this
        with ``apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)`` right
        before ``wo_a``, and the inverse rotation must use the *same* per-layer
        RoPE base that Q / KV were rotated with.

        Routed through the same fused ``apply_rope_from_positions`` path Q / KV
        use, with ``inverse=True`` negating the angle in-kernel
        (``R(-t) == R(t)^T``). Generating cos / sin inside the kernel matters
        here: this runs once per layer, so the alternative -- building cos / sin
        eagerly and broadcasting them to ``[B, S, rd/2]`` in HBM -- adds three
        small launches and a materialised tensor per layer, on all 43 of them.

        Returns a new tensor: ``attn`` is what the attention backward saved and
        must not be mutated.
        """
        if self.rotary_dim == 0:
            return attn
        rope = self.rope.get_rope(compress_ratio=self.compress_ratio)
        return apply_rope_from_positions(
            attn,
            position_ids,
            rope.inv_freq,
            rotary_dim=self.rotary_dim,
            inverse=True,
        )

    def _project_output(self, attn: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Inverse-RoPE then O projection -- the single exit for every branch.

        Kept as one helper so no attention path can reach the O projection
        without de-rotating first. Being the single exit also makes it the one
        place that reports the layer's indexer distillation loss, which has to
        happen on every layer (see :meth:`_log_indexer_distill_loss`).
        """
        self._log_indexer_distill_loss(attn.device)
        attn = self._apply_inverse_rope(attn, position_ids)
        if self.linear_o_a is not None:
            return self._grouped_o_projection(attn)
        return self._flat_o_projection(attn)

    # ------------------------------------------------------------------
    # compressed branches (HCA / CSA)
    # ------------------------------------------------------------------

    def _build_compressed_pool(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the compressor + compress-base partial RoPE.

        Returns ``[B, P, head_dim]`` where ``P = S // compress_ratio``.
        """
        device = hidden.device
        cp_group = _v4_get_cp_group()
        if cp_group is None:
            pooled = self.compressor(hidden)  # [B, P, head_dim]
        else:
            # ---- context parallel ------------------------------------------------
            # Each rank compresses only its own rows, then the pools are all-gathered
            # so every query can see the whole sequence's compressed history. Rank
            # order IS sequence order here (single BSHD sequence, S_local a multiple
            # of ratio), so no seq-major/rank-major remap is needed.
            from primus.backends.megatron.core.transformer.deepseek_v4_cp import (
                build_global_pool,
                compressor_boundary_rows,
                exchange_boundary_kv,
            )

            nb = compressor_boundary_rows(self.compress_ratio, bool(self.compressor.overlap))
            if nb > 0:
                # Overlap mode stitches window i with window i-1, which at a shard
                # boundary lives on the left neighbour. Prepend those rows, compress,
                # then drop the extra leading pool row they produced.
                bnd = exchange_boundary_kv(
                    hidden.reshape(hidden.shape[0], hidden.shape[1], 1, hidden.shape[2]),
                    nb,
                    cp_group,
                ).reshape(hidden.shape[0], nb, hidden.shape[2])
                pooled_local = self.compressor(torch.cat([bnd, hidden], dim=1))[:, 1:]
            else:
                pooled_local = self.compressor(hidden)
            pooled = build_global_pool(pooled_local, cp_group)
        B, P = pooled.shape[0], pooled.shape[1]

        # Compress-base partial RoPE. Compressed entry ``s`` stands for the
        # window starting at original token ``s * compress_ratio``, so it is
        # rotated at that position -- not at the bare block index ``s``. The
        # queries are rotated at their own original positions, so using block
        # indices here would put the two sides on different coordinate systems.
        # Matches inference/model.py, which slices ``freqs_cis[:cutoff:ratio]``
        # for prefill and indexes ``start_pos + 1 - ratio`` for decode -- both
        # land on the window's first token. Positions stay deterministic, so the
        # cached table is still reused every forward.
        cos, sin = self.rope.compress_rope.forward_arange(P, device, stride=self.compress_ratio)
        cos = cos[..., : self.rotary_dim // 2]
        sin = sin[..., : self.rotary_dim // 2]
        cos = cos.unsqueeze(0).expand(B, -1, -1)
        sin = sin.unsqueeze(0).expand(B, -1, -1)
        pool_kv = pooled.unsqueeze(2)  # [B, P, 1, head_dim]
        pool_kv = apply_interleaved_partial_rope(pool_kv, cos, sin, rotary_dim=self.rotary_dim)
        return pool_kv.squeeze(2)  # [B, P, head_dim]

    def _hca_extra_kv(
        self,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the HCA (compress_ratio == 128) compressed branch.

        Returns ``(extra_k_bh, extra_v_bh, pool, extra_mask)`` where the
        compressed pool is broadcast across H heads (single-latent
        compressor output) and the additive mask is shape ``[S, P]``
        (broadcasts over B, H). ``pool`` is the pre-broadcast ``[B, P, head_dim]``
        latent, which the caller concatenates onto the raw KV latent -- see
        :meth:`_cp_prepend_boundary` for why the concat must not happen on the
        broadcast views.

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

        extra_mask = self._hca_extra_mask_cached(S, P, device, dtype)
        return pool_bh, pool_bh, pool, extra_mask  # K = V = compressed pool

    def _hca_extra_mask_cached(self, S: int, P: int, device, dtype):
        """HCA additive causal mask ``[S, P]``, cached (data-independent).

        Pool slot ``s`` is visible to query ``t`` iff ``(s+1)*ratio - 1 <= t``;
        the mask depends only on ``(S, P, compress_ratio, dtype)`` — all fixed
        per run — so build it once instead of rebuilding arange + where every
        compressed-layer forward. Bit-identical. PRIMUS_COMPRESS_MASK_CACHE=0
        forces the eager rebuild.
        """
        cp_group = _v4_get_cp_group()
        if cp_group is not None:
            # Under CP the pool is global but the queries are this rank's slice, so the
            # visibility test must use global positions. Not cached: it depends on the
            # rank, and rebuilding an [S, P] byte mask once per layer is cheap next to
            # the attention itself.
            from primus.backends.megatron.core.transformer.deepseek_v4_cp import (
                compressed_causal_mask,
            )

            return compressed_causal_mask(
                S, P, cp_group.rank() * S, self.compress_ratio, device=device, dtype=dtype
            )
        if os.environ.get("PRIMUS_COMPRESS_MASK_CACHE", "1") == "0":
            t = torch.arange(S, device=device).unsqueeze(1)
            s_end = (torch.arange(P, device=device).unsqueeze(0) + 1) * self.compress_ratio - 1
            return torch.where(s_end <= t, 0.0, float("-inf")).to(dtype)
        cache = getattr(self, "_hca_mask_cache", None)
        if cache is None:
            cache = self._hca_mask_cache = {}
        key = (S, P, device, dtype)
        m = cache.get(key)
        if m is None:
            t = torch.arange(S, device=device).unsqueeze(1)  # [S, 1]
            s_end = (torch.arange(P, device=device).unsqueeze(0) + 1) * self.compress_ratio - 1  # [1, P]
            m = torch.where(s_end <= t, 0.0, float("-inf")).to(dtype)
            cache[key] = m
        return m

    def _csa_forward(
        self,
        hidden: torch.Tensor,
        q_bh: torch.Tensor,  # [B, H, S, head_dim]
        k_local_bh: torch.Tensor,  # [B, H, S, head_dim]
        v_local_bh: torch.Tensor,  # [B, H, S, head_dim]
        local_mask: torch.Tensor,  # [S, S] — built by caller; unused here, see below
        position_ids: Optional[torch.Tensor] = None,  # [B, S] or [S]; rotates the indexer Q
        kv: Optional[torch.Tensor] = None,  # [B, S, 1, head_dim] post-RoPE latent; CP only
    ) -> torch.Tensor:
        """CSA (compress_ratio == 4) joint local-SWA + sparse-compressed attention.

        The compressor produces a per-batch pool ``[B, P, head_dim]``,
        the indexer picks ``index_topk`` pool positions per query, and
        the attention runs softmax JOINTLY over ``[local_keys, sparse_keys]``
        so the optional ``attn_sink`` is shared across both branches.

        Plan-4 P24: the compressor / indexer / per-query top-K gather
        stay here (they are V4-specific side-paths that the kernel does
        not own); the joint-softmax math is delegated to
        :func:`primus...v4_attention_kernels._eager.reference.eager_v4_csa_attention`
        so the CSA path, the plan-4 CSA Triton kernel (P26), and the
        plan-4 unit-test harness share one definition. ``local_mask`` is
        retained in the signature for back-compat but unused — the
        reference op rebuilds the local SWA mask deterministically from
        ``swa_window`` (same call to
        :func:`sliding_window_causal_mask` as :meth:`_local_mask` makes,
        so the result is bit-identical).

        Plan-5 P31: when ``use_v4_triton_csa_attention=True`` the sparse
        top-K pool gather moves into the Triton kernel. The eager fallback
        still materialises ``gathered`` here so it remains the reference
        implementation and keeps the old P26 API covered by unit tests.
        """
        del local_mask  # see docstring
        B, H, S, Dh = q_bh.shape
        dtype = hidden.dtype

        # 0) Context parallel: like the dense and HCA branches, the raw-token sliding
        #    window straddles the shard edge, so this rank needs the left neighbour's
        #    trailing `d_window` post-RoPE KV rows. The pool half is already handled
        #    (all-gathered to global + the indexer scores against global positions).
        cp_dwindow = cp_global_start = 0
        kv_latent = None
        if _v4_get_cp_group() is not None:
            if self._csa_backend != "triton_v2":
                raise NotImplementedError(
                    "DeepSeek-V4 CSA context parallelism is only wired through the "
                    f"triton_v2 CSA backend; got '{self._csa_backend}'. The other CSA "
                    "backends build the local window themselves and would silently drop "
                    "the cross-shard part of it."
                )
            if kv is None:
                raise RuntimeError("_csa_forward needs the single-latent kv under CP")
            k_local_bh, v_local_bh, kv_latent, cp_dwindow, cp_global_start = self._cp_prepend_boundary(
                kv, B, S
            )

        # 1) Compressed pool with compress-base RoPE.
        pool = self._build_compressed_pool(hidden)  # [B, P, head_dim]
        P = pool.shape[1]

        # 2) Indexer top-K per query. The indexer is fed a *detached* hidden
        # state: it is a selector, so the only thing that should learn from its
        # distillation loss is the indexer itself. Detaching here keeps the KL
        # from leaking into the layers below (the open-source reference detaches
        # the same way), and when the indexer is frozen it also stops autograd
        # from building a subgraph whose output is discarded.
        topk_idxs, topk_scores = self.indexer(hidden.detach(), position_ids)  # [B, S, K]

        # 2b) Indexer distillation. argTopK is not differentiable and the
        # scores are otherwise discarded, so this loss is the indexer's only
        # learning signal. It is attached to ``pool`` -- which every CSA
        # backend consumes -- so the aux gradient is seeded no matter which
        # kernel the layer dispatches to, without threading a second return
        # value through all of them.
        if self.indexer_distill_enabled and self.training and torch.is_grad_enabled():
            # k_local / sink / swa_window let the target use the same joint
            # denominator the attention below does: this layer runs one softmax
            # over [window, sparse, sink], so the target for the sparse part is
            # a conditional of that, not a softmax over the sparse part alone.
            indexer_loss = compute_indexer_distill_loss(
                index_topk_scores=topk_scores,
                topk_idxs=topk_idxs,
                query=q_bh,
                pool=pool,
                softmax_scale=self._attention_scale(),
                loss_coeff=self.indexer_distill_coeff,
                head_reduce_group=self._indexer_loss_head_group(),
                k_local=k_local_bh,
                sink=self.attn_sink,
                swa_window=int(self.attn_sliding_window),
            )
            self.last_indexer_distill_loss = indexer_loss.detach()
            pool = V4IndexerLossAutoScaler.apply(pool, indexer_loss)

        # Dispatch on ``use_v4_csa_attention_backend``. gluon / triton_v2 /
        # triton_v1 consume (pool, topk) directly; eager / triton_v0 / flydsl_v0
        # use the per-query gathered [B, S, K, Dh] representation.
        be = self._csa_backend
        if be == "gluon":
            return self._v4_csa_attention_gluon(
                q_bh,
                k_local_bh,
                v_local_bh,
                pool,
                topk_idxs=topk_idxs,
                sink=self.attn_sink,
                swa_window=int(self.attn_sliding_window),
                attn_dropout=self.attn_dropout,
                training=self.training,
                scale=self._attention_scale(),
            )
        if be == "gluon_v2":
            return self._v4_csa_attention_gluon_v2(
                q_bh,
                k_local_bh,
                v_local_bh,
                pool,
                topk_idxs=topk_idxs,
                sink=self.attn_sink,
                swa_window=int(self.attn_sliding_window),
                attn_dropout=self.attn_dropout,
                training=self.training,
                scale=self._attention_scale(),
            )
        if be == "gluon_v3":
            return self._v4_csa_attention_gluon_v3(
                q_bh,
                k_local_bh,
                v_local_bh,
                pool,
                topk_idxs=topk_idxs,
                sink=self.attn_sink,
                swa_window=int(self.attn_sliding_window),
                attn_dropout=self.attn_dropout,
                training=self.training,
                scale=self._attention_scale(),
            )
        if be == "turbo":
            return self._v4_csa_attention_turbo(
                q_bh,
                k_local_bh,
                v_local_bh,
                pool,
                topk_idxs=topk_idxs,
                sink=self.attn_sink,
                swa_window=int(self.attn_sliding_window),
                attn_dropout=self.attn_dropout,
                training=self.training,
                scale=self._attention_scale(),
            )
        if be == "triton_v2":
            # Hand the kernel the un-broadcast latent when CP gave us one: it reads a
            # single key row per position anyway, and the broadcast view's gradient would
            # be an 8 GiB [B, H, Skv, D] buffer that is zero except at head 0.
            return v4_csa_attention_v2(
                q_bh,
                k_local_bh if kv_latent is None else kv_latent,
                v_local_bh if kv_latent is None else None,
                pool,
                topk_idxs=topk_idxs,
                sink=self.attn_sink,
                swa_window=int(self.attn_sliding_window),
                attn_dropout=self.attn_dropout,
                training=self.training,
                scale=self._attention_scale(),
                cp_dwindow=cp_dwindow,
                cp_global_start=cp_global_start,
                k_is_latent=kv_latent is not None,
            )
        if be == "flydsl_v1":
            return self._v4_csa_attention_flydsl(
                q_bh,
                k_local_bh,
                v_local_bh,
                pool,
                topk_idxs=topk_idxs,
                sink=self.attn_sink,
                swa_window=int(self.attn_sliding_window),
                attn_dropout=self.attn_dropout,
                training=self.training,
                scale=self._attention_scale(),
            )
        if be == "triton_v1":
            return v4_csa_attention_v1(
                q_bh,
                k_local_bh,
                v_local_bh,
                pool,
                topk_idxs=topk_idxs,
                sink=self.attn_sink,
                swa_window=int(self.attn_sliding_window),
                attn_dropout=self.attn_dropout,
                training=self.training,
                scale=self._attention_scale(),
            )

        # eager / triton_v0 / flydsl_v0: build the per-query gathered slices.
        K = topk_idxs.shape[-1]
        valid = topk_idxs >= 0  # [B, S, K]
        safe_idx = topk_idxs.clamp(min=0)
        idx_expand = safe_idx.unsqueeze(-1).expand(B, S, K, Dh)
        pool_expand = pool.unsqueeze(1).expand(B, S, P, Dh)
        gathered = torch.gather(pool_expand, dim=2, index=idx_expand)
        gathered = gathered * valid.unsqueeze(-1).to(gathered.dtype)
        sparse_mask = torch.where(valid, 0.0, float("-inf")).to(dtype)  # [B, S, K]

        if be in ("triton_v0", "flydsl_v0"):
            return v4_csa_attention_v0(
                q_bh,
                k_local_bh,
                v_local_bh,
                gathered,
                sink=self.attn_sink,
                swa_window=int(self.attn_sliding_window),
                sparse_mask=sparse_mask,
                attn_dropout=self.attn_dropout,
                training=self.training,
                scale=self._attention_scale(),
                use_flydsl=(be == "flydsl_v0"),
            )
        return eager_v4_csa_attention(
            q_bh,
            k_local_bh,
            v_local_bh,
            gathered,
            sink=self.attn_sink,
            swa_window=int(self.attn_sliding_window),
            sparse_mask=sparse_mask,
            attn_dropout=self.attn_dropout,
            training=self.training,
            scale=self._attention_scale(),
        )

    # ------------------------------------------------------------------
    # public forward
    # ------------------------------------------------------------------

    def _attention_backend_forward(
        self,
        q_bh,
        k,
        v,
        *,
        additive_mask,
        hca_local_seqlen,
        S,
        device,
        dtype,
        cp_dwindow=0,
        cp_global_start=0,
        k_latent=None,
    ):
        """Dense (cr=0) / HCA (cr=128) dispatch on ``use_v4_attention_backend``."""
        be = self._attn_backend
        if be == "gluon":
            return self._v4_attention_gluon(
                q_bh,
                k,
                v,
                sink=self.attn_sink,
                swa_window=int(self.attn_sliding_window),
                additive_mask=additive_mask,
                attn_dropout=self.attn_dropout,
                training=self.training,
                scale=self._attention_scale(),
                hca_local_seqlen=hca_local_seqlen,
            )
        if be == "gluon_v2":
            return self._v4_attention_gluon_v2(
                q_bh,
                k,
                v,
                sink=self.attn_sink,
                swa_window=int(self.attn_sliding_window),
                additive_mask=additive_mask,
                attn_dropout=self.attn_dropout,
                training=self.training,
                scale=self._attention_scale(),
                hca_local_seqlen=hca_local_seqlen,
            )
        if be == "gluon_v3":
            return self._v4_attention_gluon_v3(
                q_bh,
                k,
                v,
                sink=self.attn_sink,
                swa_window=int(self.attn_sliding_window),
                additive_mask=additive_mask,
                attn_dropout=self.attn_dropout,
                training=self.training,
                scale=self._attention_scale(),
                hca_local_seqlen=hca_local_seqlen,
            )
        if be == "turbo":
            return self._v4_attention_turbo(
                q_bh,
                k,
                v,
                sink=self.attn_sink,
                swa_window=int(self.attn_sliding_window),
                additive_mask=additive_mask,
                attn_dropout=self.attn_dropout,
                training=self.training,
                scale=self._attention_scale(),
                hca_local_seqlen=hca_local_seqlen,
            )
        if be == "triton_v2":
            # This kernel reads one key row per position (single-latent MQA), so hand it
            # the un-broadcast [B, Skv, 1, D] latent when we have it. The head-broadcast
            # view is free forward, but its gradient would be a [B, H, Skv, D] buffer that
            # is zero except at head 0 -- 8.5 GiB at 1M with CP=8. Only this backend takes
            # the latent form; the others still get the broadcast views.
            return v4_attention_v2(
                q_bh,
                k if k_latent is None else k_latent,
                v if k_latent is None else None,
                cp_dwindow=cp_dwindow,
                cp_global_start=cp_global_start,
                k_is_latent=k_latent is not None,
                sink=self.attn_sink,
                swa_window=int(self.attn_sliding_window),
                additive_mask=additive_mask,
                attn_dropout=self.attn_dropout,
                training=self.training,
                scale=self._attention_scale(),
                hca_local_seqlen=hca_local_seqlen,
            )
        if be == "flydsl_v1":
            return self._v4_attention_flydsl(
                q_bh,
                k,
                v,
                sink=self.attn_sink,
                swa_window=int(self.attn_sliding_window),
                additive_mask=additive_mask,
                attn_dropout=self.attn_dropout,
                training=self.training,
                scale=self._attention_scale(),
                hca_local_seqlen=hca_local_seqlen,
            )
        if be == "triton_v1":
            return self._attention_forward_via_v4_triton(
                q_bh,
                k,
                v,
                additive_mask,
                swa_window=int(self.attn_sliding_window),
                hca_local_seqlen=hca_local_seqlen,
            )
        # eager
        local_mask = self._local_mask(S, device=device, dtype=dtype)
        mask = local_mask if additive_mask is None else torch.cat([local_mask, additive_mask], dim=-1)
        return self._attention_forward(q_bh, k, v, mask)

    def _cp_prepend_boundary(self, kv, B, S):
        """Prepend the left neighbour's trailing window rows to the local KV.

        Every branch that runs a sliding window over RAW tokens needs this, not just
        the dense one: a query near the shard start would otherwise lose the part of
        its window that lives on the previous rank. Returns
        ``(k_bh, v_bh, kv_latent, cp_dwindow, cp_global_start)``; with CP off it returns
        the unmodified head-expanded views and ``(0, 0)``, which reproduces the non-CP
        path exactly.

        The concat happens on the SINGLE-LATENT ``[B, S, 1, D]`` tensor and the expand
        after. Concatenating the head-expanded ``[B, H, S, D]`` view instead would
        materialise a real H-fold tensor for both K and V -- 8.6 GB each at 128k rows
        with H=64 -- where the expand is otherwise free. K and V are the same tensor in
        V4's single-latent design, so one buffer serves both.

        ``kv_latent`` is that pre-expand ``[B, Skv, 1, D]`` buffer. Callers that need to
        concatenate anything else onto the key axis (HCA appends its compressed pool)
        MUST concatenate onto this and expand afterwards, for exactly the reason above:
        ``torch.cat`` on a stride-0 expanded view materialises the H-fold copy that the
        expand was avoiding.
        """
        cp_group = _v4_get_cp_group()
        if cp_group is None:
            kv_bh = kv.expand(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            return kv_bh, kv_bh, kv, 0, 0
        if self._attn_backend != "triton_v2":
            raise NotImplementedError(
                "DeepSeek-V4 context parallelism is only wired through the triton_v2 "
                f"backend (the others do not take cp_dwindow/cp_global_start); got "
                f"'{self._attn_backend}'. Set USE_V4_ATTENTION_BACKEND=triton_v2."
            )
        cp_dwindow = int(self.attn_sliding_window)
        cp_global_start = cp_group.rank() * S
        boundary_kv = _v4_exchange_boundary_kv(kv, cp_dwindow, cp_group)
        kv_full = torch.cat([boundary_kv, kv], dim=1)  # [B, d_window + S, 1, D]
        kv_full_bh = kv_full.expand(B, cp_dwindow + S, self.num_heads, self.head_dim).transpose(1, 2)
        return kv_full_bh, kv_full_bh, kv_full, cp_dwindow, cp_global_start

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

        if self.compress_ratio == 0 and self._use_core_attention:
            # Plan-3 P22: dense layer, Turbo / TE flash path.  Causal + SWA
            # are handled inside the kernel; the eager ``local_mask`` is
            # not consulted here.  KV is forwarded as ``[S, B, 1, D]``
            # and broadcast across H query heads via MQA.
            out_bh = self._attention_forward_via_core(q, kv)
            out = out_bh.transpose(1, 2).contiguous()  # [B, S, H, head_dim]
            out = out.to(dtype=dtype)
            return self._project_output(out, position_ids)

        # Broadcast K / V across the H query-head axis.
        k_h = kv.expand(B, S, self.num_heads, self.head_dim)
        v_h = kv.expand(B, S, self.num_heads, self.head_dim)

        # Move heads dim before sequence: [B, S, H, head_dim] -> [B, H, S, head_dim]
        q_bh = q.transpose(1, 2)
        k_local_bh = k_h.transpose(1, 2)
        v_local_bh = v_h.transpose(1, 2)

        if self.compress_ratio == 0:
            # ---- context parallel (dense / SWA branch) ----------------------
            # This branch is index-driven, so CP needs only the d_window post-RoPE KV rows
            # left of this shard plus the shard's global offset; the kernel is unchanged.
            # cp_dwindow == cp_global_start == 0 reproduces the non-CP path exactly.
            k_local_bh, v_local_bh, kv_latent, cp_dwindow, cp_global_start = self._cp_prepend_boundary(
                kv, B, S
            )
            out_bh = self._attention_backend_forward(
                q_bh,
                k_local_bh,
                v_local_bh,
                additive_mask=None,
                hca_local_seqlen=0,
                S=S,
                device=device,
                dtype=dtype,
                cp_dwindow=cp_dwindow,
                cp_global_start=cp_global_start,
                k_latent=kv_latent,
            )
        elif self.compress_ratio == 128:
            # HCA: the local SWA branch and the compressed-pool branch share ONE
            # softmax with ONE sink column; concatenate the pool to the local
            # keys and pass the pool-only additive mask.
            #
            # Under CP the LOCAL half needs the same left-boundary rows the dense branch
            # takes: the pool being global is not enough, because the local SWA still runs
            # over raw tokens that straddle the shard edge. The local segment then grows to
            # `cp_dwindow + S`, which is what `hca_local_seqlen` has to report -- the adapter
            # uses it as the base offset for the pool columns (`base + hca_local_seqlen + ps`),
            # so the [S, P] pool mask stays valid unchanged.
            _, _, kv_latent, cp_dwindow, cp_global_start = self._cp_prepend_boundary(kv, B, S)
            _, _, pool, extra_mask = self._hca_extra_kv(hidden)
            # Concatenate the compressed pool onto the raw KV on the SINGLE-LATENT axis,
            # then expand across heads -- the expand is a stride-0 view and costs nothing.
            # Doing it the other way round (cat on the already-broadcast [B, H, Sk, D]
            # views) materialises the H-fold copy the broadcast exists to avoid: at 1M
            # with CP=8 that is 8.51 GiB for K and another 8.51 GiB for V, per HCA layer,
            # of which the consumer reads 136 MiB -- the sparse-MLA adapter takes only
            # `k_bh[:, 0]`, and never reads `v_bh` at all (its backward returns dv=None,
            # because V4 is single-latent and the V-side gradient is structurally zero).
            # K and V are the same object here for the same reason.
            Sk = kv_latent.shape[1] + pool.shape[1]
            kv_cat = torch.cat([kv_latent, pool.unsqueeze(2)], dim=1)  # [B, Sk, 1, D]
            k_full = v_full = kv_cat.expand(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)
            out_bh = self._attention_backend_forward(
                q_bh,
                k_full,
                v_full,
                additive_mask=extra_mask,
                hca_local_seqlen=cp_dwindow + S,
                S=S,
                device=device,
                dtype=dtype,
                cp_dwindow=cp_dwindow,
                cp_global_start=cp_global_start,
                k_latent=kv_cat,
            )
        elif self.compress_ratio == 4:
            # CSA cannot use ``core_attention``: the per-query top-K
            # gather (``gathered = pool[..., topk_idxs, :]``, shape
            # ``[B, H, S, K, head_dim]``) is sparse-per-row indexed
            # attention — there is no flash-attn kernel that reads a
            # different per-query subset of keys from a pool.  Stays on
            # eager-Python under plan-3 (a custom kernel is required).
            # `_csa_forward` documents `local_mask` as retained for back-compat and
            # `del`s it on entry -- the reference op rebuilds the SWA mask from
            # `swa_window` itself. Materialising it here costs a dense [S, S] byte
            # tensor for nothing: 16 GiB at S=131072, which is what made CSA OOM at
            # 128k. Pass None; the callee never reads it.
            out_bh = self._csa_forward(hidden, q_bh, k_local_bh, v_local_bh, None, position_ids, kv)
        else:
            # Guarded by __init__; included for static-analysis completeness.
            raise ValueError(f"Unsupported compress_ratio {self.compress_ratio}")

        out = out_bh.transpose(1, 2).contiguous()  # [B, S, H, head_dim]
        out = out.to(dtype=dtype)

        return self._project_output(out, position_ids)


__all__ = [
    "DeepseekV4Attention",
    "DeepseekV4AttentionSubmodules",
]
