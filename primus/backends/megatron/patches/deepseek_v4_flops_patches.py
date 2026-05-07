###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 FLOPs reporting patch.

Plan-3 P20.  Megatron's
:func:`megatron.training.training.num_floating_point_operations` is shaped
for GPT / MLA / Mamba and gets V4 wrong on every axis that matters:

* QKV projections — V4 has a Q LoRA path
  (``hidden -> q_lora_rank -> n_heads * head_dim``) and a single-latent
  ``linear_kv`` (``hidden -> head_dim`` shared as both K and V); the
  upstream MHA/GQA branch counts a flat ``hidden * (q + k + v)`` projection
  instead.
* Output projection — V4 uses grouped low-rank ``linear_o_a`` /
  ``linear_o_b`` (``(n*d/o_groups) -> o_groups*o_lora -> hidden``); the
  upstream branch counts a flat ``(n*d) -> hidden`` proj.
* Attention scores at the wrong sequence length — V4's mHC residual
  packs ``hc_mult`` parallel streams into the layer-internal sequence axis
  (``[B, S*K, D]``).  Per-layer GEMMs run at ``S_eff = S * hc_mult``, but
  upstream uses ``args.seq_length``.
* Compressor + Indexer side paths — CSA (``compress_ratio==4``) and HCA
  (``compress_ratio==128``) layers add a Compressor (``wkv`` + ``wgate``);
  CSA additionally runs an Indexer (``w_dq`` + ``w_iuq`` + ``w_w`` +
  mini-Compressor + scoring einsum).  Upstream knows about neither.
* Hash routing — V4's first ``num_hash_layers`` MoE layers use a
  parameter-free hash router; upstream charges them the topk router GEMM
  (``hidden * num_experts``) anyway.
* MTP — V4's MTP block runs a full inner V4 transformer layer per depth
  (attention + MoE FFN) on top of the ``eh_proj``; upstream counts only
  three norms and the single ``2H -> H`` projection.

The mismatch makes per-iter MFU comparisons across PP/EP/VPP configs
meaningless because the denominator is wrong by a configuration-dependent
factor.

This module installs a single ``before_train`` patch that monkey-patches
``training_module.num_floating_point_operations`` with a wrapper.  The
wrapper:

1. Falls through to the upstream function byte-for-byte for
   ``args.model_type != "deepseek_v4"`` so dense GPT / MLA / Mamba runs
   are unchanged.
2. For V4 runs, evaluates :func:`compute_v4_flops` — a closed form
   derived from the V4 forward pass (see
   ``deepseek-v4/develop/plan-3/02-phase-details.md#p20--v4-aware-tflops-reporting``
   for the per-component derivation) — and returns its total.
3. On first invocation, logs the per-component FLOPs breakdown at rank 0
   so the formula can be sanity-checked against a hand calculation.

Convention follows upstream Megatron: pure-FMAC counts internally, then a
single ``forward_backward_factor (3) * fma_factor (2) = 6`` multiplier at
the end.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


# ---------------------------------------------------------------------------
# Shared constants (mirror Megatron's expansion factors)
# ---------------------------------------------------------------------------

_FORWARD_BACKWARD_FACTOR: int = 3  # 1 forward + 2 backward.
_FMA_FACTOR: int = 2  # multiply + add per matmul element.
_SWIGLU_FFN_EXPANSION_FACTOR: int = 3  # gate + up + down (all hidden*ffn).


# ---------------------------------------------------------------------------
# compress_ratios parsing — V4 yamls store the schedule as a JSON-like string.
# ---------------------------------------------------------------------------


def _parse_compress_ratios(raw: Any) -> Optional[List[int]]:
    """Parse ``compress_ratios`` into ``list[int]`` or return ``None``.

    Accepts ``None`` (fully dense), a string like ``"[0, 0, 4, 128]"`` (the
    YAML form), or an existing list / tuple of ints.  Mirrors the runtime
    helper in
    :func:`primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_block._parse_int_sequence`
    intentionally inline so this patch has no V4-module import dependency
    (the patch loads at ``before_train``, before model build).
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        # Tolerant of the YAML form `[0,0,4,128]` and Python repr `[0, 0, 4, 128]`.
        return [int(x) for x in ast.literal_eval(text)]
    raise TypeError(f"Unsupported compress_ratios type: {type(raw).__name__}")


def _normalize_layer_ratios(
    raw: Any,
    *,
    num_layers: int,
    mtp_num_layers: int,
) -> Tuple[List[int], List[int]]:
    """Return ``(decoder_ratios, mtp_ratios)`` each padded to its expected length.

    Mirrors the V4 block's normalization (``deepseek_v4_block._normalize_compress_ratios``)
    but also returns the trailing MTP slice when the YAML provides
    ``num_layers + mtp_num_layers`` entries (the canonical DeepSeek layout).
    Default-fills any missing slot with ``0`` (dense).
    """
    parsed = _parse_compress_ratios(raw)
    if parsed is None:
        return [0] * num_layers, [0] * mtp_num_layers

    if len(parsed) == num_layers + mtp_num_layers:
        return parsed[:num_layers], parsed[num_layers:]
    if len(parsed) == num_layers:
        return parsed, [0] * mtp_num_layers
    if len(parsed) > num_layers:
        return parsed[:num_layers], [0] * mtp_num_layers
    pad = parsed[-1] if parsed else 0
    return parsed + [pad] * (num_layers - len(parsed)), [0] * mtp_num_layers


# ---------------------------------------------------------------------------
# Per-component closed-form helpers
#
# All functions return *FMAC* counts (multiplies only).  The caller multiplies
# by ``_FMA_FACTOR * _FORWARD_BACKWARD_FACTOR = 6`` exactly once at the end.
# ---------------------------------------------------------------------------


def _attn_qkv_o_fmac_per_layer(
    *,
    batch_size: int,
    seq_len_eff: int,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    o_groups: int,
) -> int:
    """FMAC for V4 attention's projections (Q LoRA + single-latent KV + O).

    Independent of ``compress_ratio``: all V4 layer types share the same
    projection structure.  The score / softmax cost is counted separately
    by :func:`_attn_scores_fmac_per_layer`.
    """
    n_d = num_heads * head_dim

    qkv = (
        # linear_q_down_proj: hidden -> q_lora_rank
        hidden_size * q_lora_rank
        # linear_q_up_proj: q_lora_rank -> n_heads * head_dim
        + q_lora_rank * n_d
        # linear_kv (single latent): hidden -> head_dim
        + hidden_size * head_dim
    )
    if o_lora_rank > 0:
        # linear_o_a: (n*d/o_groups) -> o_groups*o_lora  ⇒ FMAC = (n*d)*o_lora
        # linear_o_b: o_groups*o_lora -> hidden          ⇒ FMAC = (o_groups*o_lora)*hidden
        o_proj = n_d * o_lora_rank + (o_groups * o_lora_rank) * hidden_size
    else:
        # Flat fallback: (n*d) -> hidden
        o_proj = n_d * hidden_size

    return batch_size * seq_len_eff * (qkv + o_proj)


def _attn_scores_fmac_per_layer(
    *,
    batch_size: int,
    seq_len_eff: int,
    num_heads: int,
    head_dim: int,
    compress_ratio: int,
    index_topk: int,
) -> int:
    """FMAC for the attention score matmuls.

    Uses Megatron's "/2 causal" convention for the local part: average key
    count per query is ``S_eff / 2``, so the (QK + PV) pair costs
    ``2 * n * d * S_eff * (S_eff / 2) = n * d * S_eff^2`` FMAC per
    micro-batch element.

    For HCA (``compress_ratio==128``) the local cost is unchanged and a
    cross-attn-style ``S_eff x P`` term is added (no causal halving on the
    pool side because the pool aggregates past tokens); for CSA
    (``compress_ratio==4``) the same local cost is paid and a sparse
    top-K cross-attn term over ``min(index_topk, P)`` keys is added.
    """
    n_d_S2 = num_heads * head_dim * seq_len_eff * seq_len_eff
    local_fmac = batch_size * n_d_S2  # local causal (QK + PV combined via /2)

    if compress_ratio == 0:
        return local_fmac

    pool = max(1, seq_len_eff // compress_ratio)

    if compress_ratio == 128:
        sparse_keys = pool  # HCA: every query attends to every pool key.
    elif compress_ratio == 4:
        sparse_keys = min(int(index_topk), pool)  # CSA top-K.
    else:
        # Forward-compatible: treat any other compressed ratio as dense+full pool.
        sparse_keys = pool

    sparse_fmac = 2 * batch_size * num_heads * seq_len_eff * sparse_keys * head_dim
    return local_fmac + sparse_fmac


def _compressor_fmac_per_layer(
    *,
    batch_size: int,
    seq_len_eff: int,
    hidden_size: int,
    head_dim: int,
    compress_ratio: int,
) -> int:
    """FMAC for the V4 :class:`Compressor` (HCA / CSA only).

    Compressor projects ``hidden -> coff*head_dim`` for both ``wkv`` and
    ``wgate``; ``coff = 2`` in overlap mode (CSA, ratio==4) and ``coff = 1``
    in non-overlap mode (HCA, ratio==128).  Inputs are at the full
    pre-pool seq length, so cost is paid at ``S_eff``.
    """
    if compress_ratio == 0:
        return 0
    coff = 2 if compress_ratio == 4 else 1
    # wkv + wgate, each hidden -> coff * head_dim
    return 2 * batch_size * seq_len_eff * hidden_size * (coff * head_dim)


def _indexer_fmac_per_layer(
    *,
    batch_size: int,
    seq_len_eff: int,
    hidden_size: int,
    compress_ratio: int,
    index_head_dim: int,
    index_n_heads: int,
) -> int:
    """FMAC for the V4 :class:`Indexer` (CSA only)."""
    if compress_ratio != 4:
        return 0

    pool = max(1, seq_len_eff // compress_ratio)
    dq_rank = index_head_dim  # Indexer.__init__ default: dq_rank = index_head_dim
    inh_ihd = index_n_heads * index_head_dim

    proj = (
        # w_dq: hidden -> dq_rank
        hidden_size * dq_rank
        # w_iuq: dq_rank -> inh * ihd
        + dq_rank * inh_ihd
        # w_w: hidden -> inh
        + hidden_size * index_n_heads
    )
    # mini-Compressor inside the Indexer: head_dim=index_head_dim, ratio=4 (coff=2),
    # so wkv + wgate each cost hidden * (2 * index_head_dim).
    mini_compressor = 2 * hidden_size * (2 * index_head_dim)
    proj += mini_compressor

    proj_fmac = batch_size * seq_len_eff * proj

    # Scoring einsum: (B, S_eff, inh, ihd) · (B, P, ihd) → (B, S_eff, inh, P)
    scoring_fmac = batch_size * seq_len_eff * index_n_heads * pool * index_head_dim

    return proj_fmac + scoring_fmac


def _moe_fmac_per_layer(
    *,
    batch_size: int,
    seq_len_eff: int,
    hidden_size: int,
    moe_ffn_hidden_size: int,
    moe_router_topk: int,
    num_experts: int,
    is_hash_layer: bool,
    shared_expert_ffn_hidden_size: int,
) -> int:
    """FMAC for the V4 MoE FFN (router + routed experts + shared expert).

    Hash-routed layers skip the topk router GEMM (they look up bucket
    indices from the raw input ids — a parameter-free op).  Non-hash
    layers pay ``hidden * num_experts`` per token for the router.
    SwiGLU's ``ffn_expansion_factor=3`` collapses the (gate + up + down)
    matmul triple into a single multiplier consistent with upstream.
    """
    router = 0 if is_hash_layer else hidden_size * num_experts
    routed = moe_router_topk * _SWIGLU_FFN_EXPANSION_FACTOR * hidden_size * moe_ffn_hidden_size
    shared = (
        _SWIGLU_FFN_EXPANSION_FACTOR * hidden_size * shared_expert_ffn_hidden_size
        if shared_expert_ffn_hidden_size > 0
        else 0
    )
    return batch_size * seq_len_eff * (router + routed + shared)


def _mtp_eh_proj_fmac(
    *,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    mtp_num_layers: int,
) -> int:
    """FMAC for the per-MTP-depth ``eh_proj`` (``2H -> H``).

    Runs at the original (un-packed) seq length because MTP runs **before**
    the V4 transformer layer's stream-lift, on the embedding output.  Two
    norms per depth are negligible.
    """
    if mtp_num_layers <= 0:
        return 0
    return mtp_num_layers * batch_size * seq_len * (2 * hidden_size) * hidden_size


def _logits_fmac(
    *,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    padded_vocab_size: int,
    mtp_num_layers: int,
) -> int:
    """FMAC for the LM head (one per main path + one per MTP depth)."""
    return (mtp_num_layers + 1) * batch_size * seq_len * hidden_size * padded_vocab_size


# ---------------------------------------------------------------------------
# Public closed form
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _V4FlopsBreakdown:
    """Per-component FMAC totals (multiply-only, pre fwd+bwd / FMA scaling)."""

    attn_qkv_o: int
    attn_scores: int
    compressor: int
    indexer: int
    moe: int
    mtp: int
    logits: int

    def total_fmac(self) -> int:
        return (
            self.attn_qkv_o
            + self.attn_scores
            + self.compressor
            + self.indexer
            + self.moe
            + self.mtp
            + self.logits
        )

    def to_total_flops(self) -> int:
        """Apply the fwd+bwd (3) × FMA (2) = 6 expansion."""
        return _FORWARD_BACKWARD_FACTOR * _FMA_FACTOR * self.total_fmac()


def compute_v4_flops(args: Any, batch_size: int) -> Tuple[int, _V4FlopsBreakdown]:
    """Closed-form V4 FLOPs for one global batch.

    Returns ``(total_flops, breakdown)`` where ``total_flops`` is the
    Megatron-convention number suitable for direct substitution into
    upstream's reporting and ``breakdown`` is a per-component report
    (FMAC pre-expansion) for sanity logging.
    """
    seq_len = int(args.seq_length)
    hc_mult = int(getattr(args, "hc_mult", 1) or 1)
    seq_len_eff = seq_len * hc_mult

    hidden_size = int(args.hidden_size)
    num_heads = int(args.num_attention_heads)
    head_dim = int(args.kv_channels)
    q_lora_rank = int(getattr(args, "q_lora_rank", 0) or 0)
    o_lora_rank = int(getattr(args, "o_lora_rank", 0) or 0)
    o_groups = int(getattr(args, "o_groups", 1) or 1)

    num_layers = int(args.num_layers)
    mtp_num_layers = int(getattr(args, "mtp_num_layers", 0) or 0)

    decoder_ratios, mtp_ratios = _normalize_layer_ratios(
        getattr(args, "compress_ratios", None),
        num_layers=num_layers,
        mtp_num_layers=mtp_num_layers,
    )

    moe_ffn_hidden_size = int(
        getattr(args, "moe_ffn_hidden_size", None) or args.ffn_hidden_size
    )
    moe_router_topk = int(getattr(args, "moe_router_topk", 1) or 1)
    num_experts = int(getattr(args, "num_experts", 1) or 1)
    shared_expert_ffn_hidden_size = int(
        getattr(args, "moe_shared_expert_intermediate_size", 0) or 0
    )
    num_hash_layers = int(getattr(args, "num_hash_layers", 0) or 0)

    index_topk = int(getattr(args, "index_topk", 0) or 0)
    index_head_dim = int(getattr(args, "index_head_dim", 0) or 0)
    index_n_heads = int(getattr(args, "index_n_heads", 0) or 0)

    padded_vocab_size = int(
        getattr(args, "padded_vocab_size", None) or args.vocab_size
    )

    # ---- decoder layers ----
    attn_qkv_o = 0
    attn_scores = 0
    compressor = 0
    indexer = 0
    moe = 0

    for layer_idx in range(num_layers):
        ratio = int(decoder_ratios[layer_idx])

        attn_qkv_o += _attn_qkv_o_fmac_per_layer(
            batch_size=batch_size,
            seq_len_eff=seq_len_eff,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            q_lora_rank=q_lora_rank,
            o_lora_rank=o_lora_rank,
            o_groups=o_groups,
        )
        attn_scores += _attn_scores_fmac_per_layer(
            batch_size=batch_size,
            seq_len_eff=seq_len_eff,
            num_heads=num_heads,
            head_dim=head_dim,
            compress_ratio=ratio,
            index_topk=index_topk,
        )
        compressor += _compressor_fmac_per_layer(
            batch_size=batch_size,
            seq_len_eff=seq_len_eff,
            hidden_size=hidden_size,
            head_dim=head_dim,
            compress_ratio=ratio,
        )
        indexer += _indexer_fmac_per_layer(
            batch_size=batch_size,
            seq_len_eff=seq_len_eff,
            hidden_size=hidden_size,
            compress_ratio=ratio,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
        )
        moe += _moe_fmac_per_layer(
            batch_size=batch_size,
            seq_len_eff=seq_len_eff,
            hidden_size=hidden_size,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            moe_router_topk=moe_router_topk,
            num_experts=num_experts,
            is_hash_layer=(layer_idx < num_hash_layers),
            shared_expert_ffn_hidden_size=shared_expert_ffn_hidden_size,
        )

    # ---- MTP layers (one full V4 layer per depth + eh_proj per depth) ----
    mtp_attn_qkv_o = 0
    mtp_attn_scores = 0
    mtp_compressor = 0
    mtp_indexer = 0
    mtp_moe = 0
    for depth in range(mtp_num_layers):
        ratio = int(mtp_ratios[depth]) if depth < len(mtp_ratios) else 0
        mtp_attn_qkv_o += _attn_qkv_o_fmac_per_layer(
            batch_size=batch_size,
            seq_len_eff=seq_len_eff,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            q_lora_rank=q_lora_rank,
            o_lora_rank=o_lora_rank,
            o_groups=o_groups,
        )
        mtp_attn_scores += _attn_scores_fmac_per_layer(
            batch_size=batch_size,
            seq_len_eff=seq_len_eff,
            num_heads=num_heads,
            head_dim=head_dim,
            compress_ratio=ratio,
            index_topk=index_topk,
        )
        mtp_compressor += _compressor_fmac_per_layer(
            batch_size=batch_size,
            seq_len_eff=seq_len_eff,
            hidden_size=hidden_size,
            head_dim=head_dim,
            compress_ratio=ratio,
        )
        mtp_indexer += _indexer_fmac_per_layer(
            batch_size=batch_size,
            seq_len_eff=seq_len_eff,
            hidden_size=hidden_size,
            compress_ratio=ratio,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
        )
        mtp_moe += _moe_fmac_per_layer(
            batch_size=batch_size,
            seq_len_eff=seq_len_eff,
            hidden_size=hidden_size,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            moe_router_topk=moe_router_topk,
            num_experts=num_experts,
            # V4's MTP depths run after num_hash_layers in the routing ordering
            # (the released checkpoint stores topk router weights for them).
            is_hash_layer=False,
            shared_expert_ffn_hidden_size=shared_expert_ffn_hidden_size,
        )

    attn_qkv_o += mtp_attn_qkv_o
    attn_scores += mtp_attn_scores
    compressor += mtp_compressor
    indexer += mtp_indexer
    moe += mtp_moe

    mtp = _mtp_eh_proj_fmac(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        mtp_num_layers=mtp_num_layers,
    )

    logits = _logits_fmac(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        padded_vocab_size=padded_vocab_size,
        mtp_num_layers=mtp_num_layers,
    )

    breakdown = _V4FlopsBreakdown(
        attn_qkv_o=attn_qkv_o,
        attn_scores=attn_scores,
        compressor=compressor,
        indexer=indexer,
        moe=moe,
        mtp=mtp,
        logits=logits,
    )
    return breakdown.to_total_flops(), breakdown


# ---------------------------------------------------------------------------
# Wrapper installation
# ---------------------------------------------------------------------------

# Module-level latch so the breakdown is logged exactly once even though
# ``num_floating_point_operations`` is called many times per training run
# (one per ``training_log`` and one per ``train_step``).
_BREAKDOWN_LOGGED = False


def _emit_breakdown(
    *,
    args: Any,
    batch_size: int,
    breakdown: _V4FlopsBreakdown,
    total_flops: int,
) -> None:
    """Emit the per-component breakdown via single-line ``log_rank_0`` calls.

    One row per ``log_rank_0`` call (instead of a single multi-line message)
    so each line passes cleanly through Primus's per-line logger formatter
    and rank-aware filter.  The header line carries the run-shape metadata
    so the breakdown can be matched to a specific ``(args, batch_size)``
    pairing in the log.
    """

    def _tflops(fmac: int) -> float:
        return fmac * _FORWARD_BACKWARD_FACTOR * _FMA_FACTOR / 1.0e12

    rows: List[Tuple[str, float]] = [
        ("attn_qkv_o", _tflops(breakdown.attn_qkv_o)),
        ("attn_scores", _tflops(breakdown.attn_scores)),
        ("compressor", _tflops(breakdown.compressor)),
        ("indexer", _tflops(breakdown.indexer)),
        ("moe", _tflops(breakdown.moe)),
        ("mtp_eh_proj", _tflops(breakdown.mtp)),
        ("logits", _tflops(breakdown.logits)),
    ]

    log_rank_0(
        "[Patch:megatron.deepseek_v4.flops_reporting] V4 closed-form FLOPs "
        f"breakdown -- batch_size={batch_size}, seq_length={int(args.seq_length)}, "
        f"hc_mult={int(getattr(args, 'hc_mult', 1) or 1)}, "
        f"num_layers={int(args.num_layers)}, "
        f"mtp_num_layers={int(getattr(args, 'mtp_num_layers', 0) or 0)}"
    )
    for name, tflops in rows:
        log_rank_0(
            f"[Patch:megatron.deepseek_v4.flops_reporting]   {name:<12s} = {tflops:9.3f} TFLOP"
        )
    log_rank_0(
        f"[Patch:megatron.deepseek_v4.flops_reporting]   {'TOTAL':<12s} = "
        f"{total_flops / 1.0e12:9.3f} TFLOP / global-batch"
    )


def _make_v4_num_floating_point_operations(original_fn, *, dispatch_v4: bool):
    """Return a wrapper that dispatches V4 vs upstream model types.

    ``dispatch_v4`` is captured at install time from ``args.model_type``
    rather than re-checked per call, because Megatron's ``pretrain()``
    overwrites ``args.model_type`` with the ``ModelType`` enum at
    ``training.py:1210`` *before* ``train()`` ever calls
    ``num_floating_point_operations``.  At that point the original
    YAML-set string ``"deepseek_v4"`` is gone and a runtime check would
    silently fall through to the upstream formula.
    """

    def wrapped(args, batch_size):
        if not dispatch_v4:
            return original_fn(args, batch_size)

        total_flops, breakdown = compute_v4_flops(args, batch_size)

        global _BREAKDOWN_LOGGED
        if not _BREAKDOWN_LOGGED:
            _BREAKDOWN_LOGGED = True
            _emit_breakdown(
                args=args,
                batch_size=batch_size,
                breakdown=breakdown,
                total_flops=total_flops,
            )

        return total_flops

    wrapped.__wrapped__ = original_fn
    wrapped._v4_flops_patched = True
    return wrapped


_TRAINER_REBIND_TARGETS: Sequence[str] = (
    # Primus's Megatron trainer imports ``num_floating_point_operations`` at
    # module load time (``primus.modules.trainer.megatron.trainer``: line 125)
    # and resolves the bare name from its OWN globals at the call site
    # (``trainer.train()``: line 1452).  Updating only
    # ``megatron.training.training.num_floating_point_operations`` therefore
    # never reaches that bound name and the trainer keeps using the upstream
    # GPT/MLA-shaped function.  We rebind the trainer's local name to the
    # wrapper too so the V4 closed form is what actually drives per-iter
    # TFLOPs reporting.  Listed explicitly so a missing module is loud rather
    # than silently silent.
    "primus.modules.trainer.megatron.trainer",
)


def _rebind_trainer_imports(wrapped_fn) -> List[str]:
    """Rebind ``num_floating_point_operations`` in every Primus module that
    captured the upstream symbol at import time.

    Returns the list of modules that were actually rebound so the install log
    can show whether each downstream binding was wired up.  Modules that were
    not yet imported (e.g. on a cold-cache trainer init) are skipped silently
    — the trainer module imports the function at its own load, which
    happens before ``before_train``, so in practice the targeted module is
    always present at this point.
    """
    import sys

    rebound: List[str] = []
    for module_name in _TRAINER_REBIND_TARGETS:
        mod = sys.modules.get(module_name)
        if mod is None:
            continue
        if getattr(mod, "num_floating_point_operations", None) is wrapped_fn:
            continue
        if not hasattr(mod, "num_floating_point_operations"):
            continue
        mod.num_floating_point_operations = wrapped_fn
        rebound.append(module_name)
    return rebound


@register_patch(
    "megatron.deepseek_v4.flops_reporting",
    backend="megatron",
    phase="before_train",
    description=(
        "DeepSeek-V4: replace Megatron's GPT/MLA-shaped "
        "num_floating_point_operations with a V4 closed form (Q LoRA + "
        "single-latent KV + grouped low-rank O at S * hc_mult, plus "
        "Compressor / Indexer side paths and MTP per-depth full inner "
        "layer cost). Falls through byte-for-byte for non-V4 model types."
    ),
    condition=lambda ctx: getattr(get_args(ctx), "model_type", None) == "deepseek_v4",
)
def patch_v4_flops_reporting(ctx: PatchContext):
    """Install the V4 FLOPs wrapper on ``training.num_floating_point_operations``."""
    import megatron.training.training as training_module

    original_fn = training_module.num_floating_point_operations
    if getattr(original_fn, "_v4_flops_patched", False):
        log_rank_0(
            "[Patch:megatron.deepseek_v4.flops_reporting] "
            "num_floating_point_operations already patched, skip"
        )
        return

    # Capture the V4 dispatch decision NOW, while ``args.model_type`` is
    # still the YAML-set string.  Megatron's ``pretrain()`` will rebind
    # ``args.model_type`` to a ``ModelType`` enum at
    # ``training.py:1210`` later, so any runtime check inside the wrapper
    # would fail.  This patch is gated to only install when the YAML model
    # type is V4 (see ``condition`` on the decorator), so it's safe to
    # hard-set ``dispatch_v4=True`` here.
    wrapped = _make_v4_num_floating_point_operations(original_fn, dispatch_v4=True)
    training_module.num_floating_point_operations = wrapped
    rebound = _rebind_trainer_imports(wrapped)

    log_rank_0(
        "[Patch:megatron.deepseek_v4.flops_reporting] wrapped "
        "num_floating_point_operations; per-iter TFLOPs now reported "
        "with V4-aware closed form (see "
        "deepseek-v4/develop/plan-3/02-phase-details.md#p20 for the formula)."
    )
    if rebound:
        log_rank_0(
            "[Patch:megatron.deepseek_v4.flops_reporting] rebound trainer "
            f"import bindings: {rebound}"
        )
    else:
        log_rank_0(
            "[Patch:megatron.deepseek_v4.flops_reporting] no trainer modules "
            "needed rebinding (none of "
            f"{list(_TRAINER_REBIND_TARGETS)} were imported yet)."
        )


__all__: Sequence[str] = (
    "compute_v4_flops",
    "patch_v4_flops_reporting",
)
