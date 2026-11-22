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

try:  # PRIMUS_OVERLAP_INDEXER (side-stream CSA indexer); absent in older Primus-Turbo builds
    from primus_turbo.pytorch.core import overlap as _overlap
except ImportError:  # pragma: no cover
    _overlap = None
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
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.q_norm_rope import (
    q_norm_rope_pad,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.qkv_post import (
    qkv_post,
    qkv_post_supported,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.rope_interleaved_partial import (
    apply_rope_from_positions,
    orope_inplace_supported,
    out_inv_rope_inplace,
)

_SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)

logger = logging.getLogger(__name__)


def _v4_get_cp_group():
    from primus.backends.megatron.core.transformer.deepseek_v4_cp import get_cp_group

    return get_cp_group()


def _thd_pool_visibility(S, P, cu_seqlens, ratio, pool_identity, cp_group, device, dtype):
    """``[S, P]`` additive mask: which compressed slots a packed query may attend to.

    Three conditions, all necessary:
      * the slot is used at all (``comp_id >= 0``; the compact layout pads to a fixed
        capacity so the all-gather stays uniform across ranks),
      * it belongs to the query's OWN packed sequence -- otherwise one sample conditions
        on another, silently,
      * it is already complete at the query's position: slot ``k`` of a sequence covers
        that sequence's rows ``[k*ratio, (k+1)*ratio)``, so it needs
        ``(k+1)*ratio - 1 <= u`` for local query position ``u``.

    Shared by the attention and the indexer on purpose: the indexer's top-K addresses
    exactly these slots, so any disagreement shows up as selected-but-invisible columns.
    """
    cu = cu_seqlens.to(device=device, dtype=torch.int64)
    if P == 0:
        return torch.zeros(S, 0, device=device, dtype=dtype)
    seq_of_pool, comp_of_pool = pool_identity

    # Queries are this rank's rows; the pool is global after the all-gather. Work in
    # global row coordinates and map this rank's rows in with global_start.
    global_start = 0 if cp_group is None else cp_group.rank() * S
    n_rows = (cu[1:] - cu[:-1]).to(torch.int64)
    seq_of_row = torch.repeat_interleave(torch.arange(n_rows.numel(), device=device), n_rows)[
        global_start : global_start + S
    ]
    local_q = (torch.arange(S, device=device) + global_start) - cu[:-1][seq_of_row]

    used = comp_of_pool.unsqueeze(0) >= 0
    same = seq_of_row.unsqueeze(1) == seq_of_pool.unsqueeze(0)
    causal = (comp_of_pool.unsqueeze(0) + 1) * ratio - 1 <= local_q.unsqueeze(1)
    return torch.where(used & same & causal, 0.0, float("-inf")).to(dtype)


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


# ---------------------------------------------------------------------------
# wq_a || wkv concat (PRIMUS_V4_QKV_A_CONCAT=1)
# ---------------------------------------------------------------------------
# linear_q_down_proj (wq_a, D -> q_lora_rank) and linear_kv (wkv, D -> head_dim) read the same
# hidden. With the switch on they run as ONE GEMM against W = cat([wq_a, wkv, 0-pad]) [Npad, D]:
#   fwd   one x quant (1x128) + one GEMM N=Npad; q_a / kv are strided views of the output
#   dgrad one GEMM with K=Npad (dX accumulated in fp32 once instead of two bf16 GEMMs + a bf16 add)
#   wgrad one [Npad, D] GEMM; cat's backward slices it into both parameters' .grad
# Npad = round_up(nq + nkv, PRIMUS_V4_QKV_A_CONCAT_PAD) (default 512 -> 2048 for 1024+576: passes the
# g90 FP8 guard N%512 and its dgrad K%512; 128 -> 1664 for backends that take N%128 and any K%128).
# Quantization is identical to the separate path (wq_a rows end on a 128-block boundary, the zero pad
# leaves wkv's last 128x128 block amax unchanged; 1x128 dY groups line up the same way) -- see
# ~/dsv41/work/opt3/small/splitk/cpu_check_small.py. PRIMUS_V4_QKV_A_CONCAT_VIEWS=0 makes q_a / kv
# contiguous copies (only needed if a consumer rejects row-strided inputs).
# densefp8 (2026-11-21): this path was never active in the DSv4 proxy -- TE-style linears keep ``bias`` as an empty
# tensor, which the old ``bias is not None`` check rejected. Now:
#   PRIMUS_V4_QKV_A_CONCAT_FP8BLK=1 -> Turbo gemm_fp8_block_concat_n on the block-FP8 dsv4up path (joint FP8 weight
#                                       written by the optimizer, fused main_grad wgrad, order [wkv, wq_a]);
#   otherwise the separate linears (as before), unless PRIMUS_V4_QKV_A_CONCAT_LEGACY=1 asks for the generic concat.
def _qkv_a_concat_enabled() -> bool:
    return os.environ.get("PRIMUS_V4_QKV_A_CONCAT", "0") == "1"


def _no_bias(m: nn.Module) -> bool:
    """No bias: TE-style linears keep ``bias`` as an EMPTY tensor when bias=False (not None)."""
    if getattr(m, "use_bias", False):
        return False
    b = getattr(m, "bias", None)
    return b is None or (isinstance(b, torch.Tensor) and b.numel() == 0)


class _QKVASplit(torch.autograd.Function):
    """[.., Npad] -> ([.., nq], [.., nkv]) views; backward writes ONE [.., Npad] grad (zero pad)."""

    @staticmethod
    def forward(ctx, out, nq: int, nkv: int):  # type: ignore[override]
        ctx.nq, ctx.nkv, ctx.shape = nq, nkv, out.shape
        views = os.environ.get("PRIMUS_V4_QKV_A_CONCAT_VIEWS", "1") == "1"
        q = out.narrow(-1, 0, nq)
        kv = out.narrow(-1, nq, nkv)
        if not views:
            q, kv = q.contiguous(), kv.contiguous()
        return q, kv

    @staticmethod
    def backward(ctx, dq, dkv):  # type: ignore[override]
        nq, nkv = ctx.nq, ctx.nkv
        ref = dq if dq is not None else dkv
        g = torch.empty(ctx.shape, dtype=ref.dtype, device=ref.device)
        if dq is None:
            g.narrow(-1, 0, nq).zero_()
        else:
            g.narrow(-1, 0, nq).copy_(dq)
        if dkv is None:
            g.narrow(-1, nq, nkv).zero_()
        else:
            g.narrow(-1, nq, nkv).copy_(dkv)
        if ctx.shape[-1] > nq + nkv:
            g.narrow(-1, nq + nkv, ctx.shape[-1] - nq - nkv).zero_()
        return g, None, None


def _qkv_a_fp8blk(x2: torch.Tensor, ws, config):
    """PRIMUS_V4_QKV_A_CONCAT on the Turbo block-FP8 dsv4up path: one x quant + one launch per pass against the
    joint FP8 weight the optimizer writes (Turbo gemm_fp8_block_concat_n; fused main_grad wgrad, no bf16 weight
    cat, no .grad round trip). Returns one contiguous [M, N_i] output per weight of ``ws``, or None when that
    path does not apply."""
    if os.environ.get("PRIMUS_V4_QKV_A_CONCAT_FP8BLK", "0") != "1":
        return None
    try:
        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboLowPrecisionGlobalStateManager as _M,
        )
        from primus.backends.megatron.core.extensions.primus_turbo import (
            _dense_precision_context,
            _fuse_wgrad_accum_pattern,
        )
        from primus_turbo.pytorch.ops.gemm_fp8_concat import (
            gemm_fp8_block_concat_n,
            gemm_fp8_block_concat_n_supported,
        )
    except Exception:  # CPU unit tests / a Turbo without the op
        return None
    with _dense_precision_context(x2, ws[0]):
        if not _M.is_turbo_fp8_enabled():
            return None
        qc = _M.get_turbo_quant_config()
        if qc is None or not qc.block_scaling():
            return None
        cfg = qc.data()
        if not gemm_fp8_block_concat_n_supported(x2, ws, cfg):
            return None
        pats = [_fuse_wgrad_accum_pattern(config, w) for w in ws]
        return gemm_fp8_block_concat_n(x2, list(ws), cfg, pats)


def _concat_linear(x2: torch.Tensor, w: torch.Tensor, ref_module: nn.Module) -> torch.Tensor:
    """y = x2 @ w^T through the same Turbo precision path a PrimusTurbo*Linear would take for this weight."""
    try:
        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboLowPrecisionGlobalStateManager as _M,
        )
        from primus.backends.megatron.core.extensions.primus_turbo import (
            _dense_precision_context,
        )
    except Exception:  # CPU unit tests without Turbo
        return x2 @ w.t()
    with _dense_precision_context(x2, w):
        if _M.is_turbo_fp8_enabled():
            import primus_turbo.pytorch as pt

            qc = _M.get_turbo_quant_config()
            assert qc.block_scaling() or qc.mxfp8_scaling() or qc.current_scaling(), "qkv_a concat: unsupported fp8 recipe"
            # .grad path (no fused main_grad accumulate): cat's backward splits dW into both params.
            return pt.ops.gemm_fp8(x2, w, trans_a=False, trans_b=True, out_dtype=None, config=qc.data())
        return x2 @ w.t()


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


_OA_DGRAD_TN = os.environ.get("PRIMUS_V4_OA_DGRAD_TN", "0") == "1"
# bf16g2: o_a dgrad as an NN Gluon GEMM on w itself (no [G, d, r] transpose of w); bit-identical to the hipBLASLt
# TN / NN solutions (same per-output K order). Needs Triton main (Gluon cdna5); falls back to the paths below.
_BF16G2_OA_DGRAD = os.environ.get("PRIMUS_BF16G2_OA_DGRAD", "0") == "1"
_BF16G2_STATE = {}


def _bf16g2_mm_or_none():
    """primus_turbo bf16g2_mm if importable (Gluon cdna5 / gfx1250), else None (logged once)."""
    if "mm" not in _BF16G2_STATE:
        try:
            from primus_turbo.gluon.gemm.bf16g2_gemm import bf16g2_mm

            _BF16G2_STATE["mm"] = bf16g2_mm
        except Exception as e:  # noqa: BLE001
            _BF16G2_STATE["mm"] = None
            logging.getLogger(__name__).warning(f"bf16g2 unavailable ({type(e).__name__}: {e}); using hipBLASLt")
    return _BF16G2_STATE["mm"]


class _GroupedOAProjFn(torch.autograd.Function):
    """Grouped-O ``o_a`` down-projection ``[B,S,G,d] x [G,r,d] -> [B,S,G*r]`` without layout copies.

    ``torch.einsum("bsgd,grd->bsgr")`` runs a bmm over ``g`` on permuted views and pays
    for it in layout copies: the ``[G,BS,r]`` result is materialised as ``[BS,G,r]`` for
    the ``flatten`` (fwd), the input gradient comes back as ``[G,BS,d]`` and is copied to
    ``[BS,G,d]`` by ``ViewBackward`` of the ``reshape``, and the weight gradient comes back
    as ``[G,d,r]`` and is copied to ``[G,r,d]`` by ``ViewBackward`` of ``weight.view`` --
    three full transposing copies per layer (~0.3 ms/layer on MI455X at S=4096).

    Here every bmm reads and writes through strided batch views (hipBLASLt takes any
    batch stride with a unit-stride inner dim), so the outputs land directly in the
    layout the neighbours want: same GEMMs, no copies.
    """

    @staticmethod
    def forward(ctx, x, w, wview=None):  # type: ignore[override]
        # x: [B, S, G, d] (rows unit-stride), w: [G, r, d]
        # wview = (G, r, d): ``w`` is the 2-D weight Parameter itself and its dW goes straight into ``main_grad``
        # (PRIMUS_V4_OA_MAIN_GRAD=1, see _oa_main_grad_ok) instead of param.grad + the DDP hook's copy / add.
        ctx.wparam, ctx.wview = None, wview
        if wview is not None:
            ctx.wparam = w
            w.grad_added_to_main_grad = True  # set in forward, like Turbo's fused-wgrad linears
        wv = w.view(wview) if wview is not None else w
        B, S, G, d = x.shape
        r = wv.shape[1]
        x2 = x.reshape(B * S, G, d)
        out = x.new_empty(B * S, G, r)
        torch.bmm(x2.transpose(0, 1), wv.transpose(1, 2), out=out.transpose(0, 1))
        ctx.save_for_backward(x2, w)
        ctx.x_shape = x.shape
        return out.view(B, S, G * r)

    @staticmethod
    def backward(ctx, dout):  # type: ignore[override]
        x2, w = ctx.saved_tensors
        wshape = w.shape
        if ctx.wview is not None:
            w = w.view(ctx.wview)
        BS, G, d = x2.shape
        r = w.shape[1]
        do = dout.reshape(BS, G, r).transpose(0, 1)  # [G, BS, r] strided view
        dx = None
        if ctx.needs_input_grad[0]:
            dx = x2.new_empty(BS, G, d)
            g2 = _bf16g2_mm_or_none() if _BF16G2_OA_DGRAD else None
            if (g2 is not None and w.stride(-1) == 1 and do.stride(-1) == 1 and w.dtype == torch.bfloat16
                    and do.dtype == torch.bfloat16 and r % 128 == 0 and r >= 384):
                # NN: B = w [G, r(K), d(N)], N contiguous; 256x256 tiles, 16-element row pad for the MN-major tile
                g2(do, w, dx.transpose(0, 1), tile="256x256", tpad=16)
                if not _BF16G2_STATE.get("logged"):
                    _BF16G2_STATE["logged"] = True
                    print(f"[bf16g2] o_a dgrad NN Gluon (G={G}, T={BS}, r={r}, d={d})", flush=True)
            elif _OA_DGRAD_TN and w.stride(-1) == 1:
                # TN: K (= r) contiguous for both operands. With the TunableOp entry for
                # tn_4096_4096_1024_B_8_ld_1024_8192_32768 (tunableop_dsv4_gfx1250_bf16{bit,all}_0.csv) hipBLASLt
                # runs ~20% faster than the tuned NN form and bit-identical to it; without the entry the default
                # TN kernel is SLOWER. The [G, d, r] copy of w costs ~13 us (Triton, LDS-tiled).
                from primus.backends.megatron.core.fusions.bf16_transpose import (
                    transpose_last2,
                )

                torch.bmm(do, transpose_last2(w).transpose(1, 2), out=dx.transpose(0, 1))
            else:
                torch.bmm(do, w, out=dx.transpose(0, 1))
            dx = dx.view(ctx.x_shape)
        dw = None
        if ctx.needs_input_grad[1]:
            if ctx.wparam is not None:
                from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad, main_grad_first_write

                mg = ctx.wparam.main_grad.view(G, r, d)
                if main_grad_first_write(ctx.wparam):  # beta 0: the GEMM writes main_grad (= param.grad's bits)
                    torch.bmm(do.transpose(1, 2), x2.transpose(0, 1), out=mg)
                else:  # the DDP hook's add, done here
                    mg.add_(torch.bmm(do.transpose(1, 2), x2.transpose(0, 1)))
                dw = _get_dummy_wgrad(list(wshape), w.dtype)
                ctx.wparam = None
            else:
                dw = torch.bmm(do.transpose(1, 2), x2.transpose(0, 1))  # [G, r, d], contiguous
        return dx, dw, None


def _oa_main_grad_ok(weight: torch.Tensor) -> bool:
    """PRIMUS_V4_OA_MAIN_GRAD=1 (default off): the grouped o_a bmm writes its dW into ``weight.main_grad``."""
    if os.environ.get("PRIMUS_V4_OA_MAIN_GRAD", "0") != "1":
        return False
    mg = getattr(weight, "main_grad", None)
    return (
        isinstance(weight, torch.nn.Parameter)
        and hasattr(weight, "grad_added_to_main_grad")
        and isinstance(mg, torch.Tensor)
        and mg.shape == weight.shape
        and mg.dtype == weight.dtype
        and mg.is_contiguous()
        and weight.is_contiguous()
        and not getattr(weight, "shared", False)
    )


def _glue_wqb_dyq_ok(linear) -> bool:
    """PRIMUS_GLUE_WQB_DYQ=1 (default off): the wq_b linear (no bias, TP=1) may take its dY pre-quantized from the
    fused q norm/rope backward. The slot only opens when that linear really runs the Turbo g90/dsv4up block FP8 path
    with the FUSEQ quant layouts; otherwise nothing changes."""
    if os.environ.get("PRIMUS_GLUE_WQB_DYQ", "0") != "1" or os.environ.get("PRIMUS_TURBO_FUSEQ", "0") != "1":
        return False
    b = getattr(linear, "bias", None)
    if isinstance(b, torch.Tensor) and b.numel() > 0:
        return False
    if getattr(getattr(linear, "config", None), "tensor_model_parallel_size", 1) != 1:
        return False
    try:
        from primus_turbo.pytorch.ops.gemm_fp8 import request_dy_prequant  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _grouped_o_a_bmm_enabled() -> bool:
    return os.environ.get("PRIMUS_V4_GROUPED_OA_BMM", "1") != "0"


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

    def _qkv_a_concat_ok(self, hidden: torch.Tensor) -> bool:
        if not _qkv_a_concat_enabled():
            return False
        wq = getattr(self.linear_q_down_proj, "weight", None)
        wkv = getattr(self.linear_kv, "weight", None)
        if wq is None or wkv is None or wq.dim() != 2 or wkv.dim() != 2 or wq.shape[1] != wkv.shape[1]:
            return False
        if not (_no_bias(self.linear_q_down_proj) and _no_bias(self.linear_kv)):
            return False
        return getattr(self.config, "tensor_model_parallel_size", 1) == 1 and hidden.shape[-1] == wq.shape[1]

    def _fuse2_att_gadd_slot(self, hidden: torch.Tensor, seq_starts):
        """PRIMUS_FUSE2_ATT_GADD=1 (fuse2, default off): a gradient slot when this compressed layer's input feeds only
        the concat GEMM and the compressor's main_grad-writing wkv_gate (the indexer reads ``hidden.detach()``)."""
        from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.fuse2_main_grad import (
            AttGradSlot,
            att_gadd_enabled,
            fuse2_linear_mg_ok,
        )

        if not att_gadd_enabled() or self.compress_ratio not in (4, 128) or seq_starts is not None:
            return None
        if getattr(self, "_ovl_pool", None) is not None:  # PRIMUS_OVERLAP_COMPRESSOR already issued it (ForkGrad sums)
            return None
        comp = getattr(self, "compressor", None)
        wkg = getattr(comp, "wkv_gate", None) if comp is not None else None
        if wkg is None or getattr(wkg, "bias", None) is not None or not getattr(comp, "_fuse_proj", False):
            return None
        if not (torch.is_grad_enabled() and hidden.requires_grad and hidden.dtype == torch.bfloat16
                and hidden.is_contiguous() and _v4_get_cp_group() is None and fuse2_linear_mg_ok(hidden, wkg.weight)):
            return None
        return AttGradSlot()

    def _qkv_a_concat(self, hidden: torch.Tensor):
        """One GEMM for wq_a and wkv -> (q_compressed [B,S,nq], kv_proj [B,S,nkv]) (row-strided views)."""
        wq = self.linear_q_down_proj.weight
        wkv = self.linear_kv.weight
        nq, nkv, D = wq.shape[0], wkv.shape[0], wq.shape[1]
        lead = hidden.shape[:-1]
        # Block-FP8 dsv4up path (the DSv4 recipe): joint FP8 weight, fused main_grad. The weight order follows the
        # main_grad layout (DDP lays wkv right before wq_a), so the wgrad is one GEMM into both main_grads; it is
        # fixed at the first call (the optimizer's joint FP8 cache is built in that order).
        order = getattr(self, "_qkv_a_fp8_order", None)
        if order is None:
            mq, mk = getattr(wq, "main_grad", None), getattr(wkv, "main_grad", None)
            kv_first = (
                isinstance(mq, torch.Tensor)
                and isinstance(mk, torch.Tensor)
                and mk.data_ptr() + mk.numel() * mk.element_size() == mq.data_ptr()
            )
            order = "kv_q" if kv_first else "q_kv"
            self._qkv_a_fp8_order = order
            if os.environ.get("PRIMUS_V4_QKV_A_CONCAT_DEBUG", "0") == "1" and isinstance(mq, torch.Tensor) and isinstance(mk, torch.Tensor):
                print(f"[qkv_a concat] order={order} wq.main_grad ptr={mq.data_ptr():#x} n={mq.numel()} "
                      f"wkv.main_grad ptr={mk.data_ptr():#x} n={mk.numel()} es={mq.element_size()}", flush=True)
        ws = (wkv, wq) if order == "kv_q" else (wq, wkv)
        x2 = hidden.reshape(-1, D)
        outs = _qkv_a_fp8blk(x2, ws, self.config)
        if outs is not None:  # one contiguous output per weight, in ``ws`` order
            o_q, o_kv = (outs[1], outs[0]) if order == "kv_q" else (outs[0], outs[1])
            return o_q.view(*lead, nq), o_kv.view(*lead, nkv)
        if os.environ.get("PRIMUS_V4_QKV_A_CONCAT_LEGACY", "0") != "1":
            # The generic concat below (bf16 weight cat + per-call weight quant, .grad wgrad) was never active in
            # the DSv4 proxy (the TE empty-bias check rejected it); keep the separate linears unless asked.
            return None, None
        pad_to = int(os.environ.get("PRIMUS_V4_QKV_A_CONCAT_PAD", "512"))
        npad = (nq + nkv + pad_to - 1) // pad_to * pad_to
        parts = [wq, wkv]
        if npad > nq + nkv:
            zp = getattr(self, "_qkv_a_zero_pad", None)
            if zp is None or zp.shape != (npad - nq - nkv, D) or zp.dtype != wq.dtype or zp.device != wq.device:
                zp = torch.zeros(npad - nq - nkv, D, dtype=wq.dtype, device=wq.device)
                self._qkv_a_zero_pad = zp
            parts.append(zp)
        w = torch.cat(parts, dim=0)  # [Npad, D]; one bf16 copy (an Adam-fused weight quant would emit it directly)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        out = _concat_linear(x2, w, self.linear_q_down_proj).view(*lead, npad)
        return _QKVASplit.apply(out, nq, nkv)

    def _apply_q(self, hidden: torch.Tensor, q_compressed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``[B, S, D]`` → ``[B, S, H, head_dim]`` (Q after q_norm + q_rms)."""
        if q_compressed is None:
            q_compressed = _projection_forward(self.linear_q_down_proj, hidden)
        q_compressed = self.q_layernorm(q_compressed)
        q = _projection_forward(self.linear_q_up_proj, q_compressed)
        B, S, _ = q.shape
        q = q.view(B, S, self.num_heads, self.head_dim)
        # Per-head parameter-less RMS (matches `inference/model.py`).
        q = _per_head_rms_norm(q, eps=self.norm_eps)
        return q

    def _fused_q_norm_rope_ok(self, hidden: torch.Tensor) -> bool:
        if os.environ.get("PRIMUS_V4_Q_NORM_ROPE_FUSED", "1") == "0":
            return False
        if not (hidden.is_cuda and self.rotary_dim > 0):
            return False
        if os.environ.get("PRIMUS_ROPE_TRITON", "1") == "0" or os.environ.get("PRIMUS_RMSNORM_TRITON", "1") == "0":
            return False
        hd = int(self.head_dim)
        return hd >= 2 and (hd & (hd - 1)) == 0 and hidden.dtype in (torch.bfloat16, torch.float16)

    def _apply_q_norm_rope(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        q_compressed: Optional[torch.Tensor] = None,
        q_normed: bool = False,
    ) -> torch.Tensor:
        """``_apply_q`` + Q RoPE in one fused kernel (same math, see ``q_norm_rope``). ``q_normed``: ``q_compressed``
        already went through ``q_layernorm`` (PRIMUS_V4_QKV_POST_FUSED)."""
        if q_compressed is None:
            q_compressed = _projection_forward(self.linear_q_down_proj, hidden)
        if not q_normed:
            q_compressed = self.q_layernorm(q_compressed)
        dyq_slot = None
        if _glue_wqb_dyq_ok(self.linear_q_up_proj):
            # PRIMUS_GLUE_WQB_DYQ=1: the wq_b FP8 GEMM opens a dY-prequant slot; the q norm/rope backward fills it
            # with the FP8 dY copies, so wq_b's backward skips its own quant pass (Turbo DyPrequantSlot).
            from primus_turbo.pytorch.ops.gemm_fp8 import request_dy_prequant, take_dy_prequant_slot

            request_dy_prequant()
            try:
                q = _projection_forward(self.linear_q_up_proj, q_compressed)
            finally:
                dyq_slot = take_dy_prequant_slot()
        else:
            q = _projection_forward(self.linear_q_up_proj, q_compressed)
        B, S, _ = q.shape
        q = q.view(B, S, self.num_heads, self.head_dim)
        rope = self.rope.get_rope(compress_ratio=self.compress_ratio)
        pos_flat = position_ids.broadcast_to((B, S)).reshape(-1)
        # The triton_v2 sparse-MLA adapter wants the query zero-padded by 64 columns.
        be = self._csa_backend if self.compress_ratio == 4 else self._attn_backend
        pad = 64 if (be == "triton_v2" and not (self.compress_ratio == 0 and self._use_core_attention)) else 0
        return q_norm_rope_pad(
            q, pos_flat, rope.inv_freq, rotary_dim=self.rotary_dim, eps=self.norm_eps, pad=pad, dyq_slot=dyq_slot
        )

    def _apply_kv(self, hidden: torch.Tensor, kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``[B, S, D]`` → ``[B, S, 1, head_dim]`` (single-latent K = V)."""
        if kv is None:
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

    def _local_mask(self, S: int, *, device, dtype, seq_starts=None) -> torch.Tensor:
        """Mask for the local (SWA or full causal) branch.

        ``attn_sliding_window > 0`` enables sliding-window; ``0`` (the
        default for unit tests / configs without SWA) gives full causal.

        ``seq_starts`` (``[S]``, from :meth:`_thd_seq_starts`) additionally blocks
        attention across packed-sequence boundaries: a query may only see keys at or
        after its OWN sequence's first row. Without it a packed batch is just one long
        causal sequence and every sample is silently conditioned on its predecessors.
        """
        window = self.attn_sliding_window if self.attn_sliding_window > 0 else 0
        mask = sliding_window_causal_mask(S, window if window > 0 else S, device=device, dtype=dtype)
        if seq_starts is not None:
            starts = seq_starts.to(device=device, dtype=torch.long)
            keys = torch.arange(S, device=device).unsqueeze(0)  # [1, Sk]
            same_seq = keys >= starts.unsqueeze(1)  # [Sq, Sk]
            mask = mask.masked_fill(~same_seq, float("-inf"))
        return mask

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
        (``B=1, H=64, S=4096``) that's 4 GiB / microbatch, and the
        dominant activation cost.  ``hc_mult`` does not enter here: the
        mHC streams are collapsed to one hidden per token before this
        runs (``DeepseekV4HybridLayer._hc_apply``).

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
                o = _fp8_grouped_o_a(attn_g, wo_a_w).flatten(2)  # per-group MXFP8
            elif _grouped_o_a_bmm_enabled() and attn_g.stride(-1) == 1 and attn_g.is_cuda:
                if _oa_main_grad_ok(weight):
                    weight._primus_fwd_claim = True  # grad_added_to_main_grad is flagged already in forward
                    o = _GroupedOAProjFn.apply(attn_g, weight, tuple(wo_a_w.shape))
                else:
                    o = _GroupedOAProjFn.apply(attn_g, wo_a_w)  # [B, S, G*r], no layout copies
            else:
                o = torch.einsum("bsgd,grd->bsgr", attn_g, wo_a_w).flatten(2)
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
        if orope_inplace_supported(attn, self.rotary_dim) and torch.is_grad_enabled() and attn.requires_grad:
            # PRIMUS_V4_OROPE_INPLACE=1: ``attn`` is the sparse-MLA output its backward saved; de-rotate its tail in
            # place and restore it in the backward (same bytes as below; see OutInvRopeInplaceFn).
            from primus.backends.megatron.core.transformer.v4_attention_kernels.v4_sparse_mla_adapter import (
                take_saved_output,
            )

            if take_saved_output(attn):
                return out_inv_rope_inplace(attn, position_ids, rope.inv_freq, rotary_dim=self.rotary_dim)
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

    def _build_compressed_pool(self, hidden: torch.Tensor, cu_seqlens=None) -> torch.Tensor:
        """Run the compressor + compress-base partial RoPE.

        Returns ``[B, P, head_dim]`` where ``P = S // compress_ratio``.

        Under packing (``cu_seqlens`` given) the pool is the concatenation of each
        sequence's own windows, and the compress-base RoPE position restarts at 0 for
        every sequence -- pool slot k of sequence i must carry phase k, not phase
        (global slot index), or every sequence after the first is rotated wrongly.
        """
        device = hidden.device
        cp_group = _v4_get_cp_group()
        if cu_seqlens is not None:
            # No alignment requirement: a window may straddle a shard boundary and the
            # exchange below supplies the rows owned by the previous rank. Alignment
            # used to stand in for that, at the cost of 51.5% of tokens being padding
            # (34.2% supervised, against 56.1% unaligned -- measured over every pack).
            # The compressor sees only this rank's rows, so it needs LOCAL boundaries;
            # the RoPE phase below is per-sequence and therefore also local.
            S_loc = hidden.shape[1]
            gstart = 0 if cp_group is None else cp_group.rank() * S_loc
            cu_seqlens if cp_group is None else self._thd_local_cu(cu_seqlens, gstart, S_loc)
            boundary = None
            if cp_group is not None:
                from primus.backends.megatron.core.transformer.deepseek_v4_cp import (
                    compressor_boundary_rows,
                    exchange_boundary_hidden,
                )

                d_win = compressor_boundary_rows(self.compress_ratio, bool(self.compressor.overlap))
                boundary = exchange_boundary_hidden(hidden, d_win, cp_group)
            pooled = self.compressor(
                hidden,
                cu_seqlens=cu_seqlens,
                global_start=gstart,
                boundary_hidden=boundary,
            )
            P = pooled.shape[1]
            # The compress-base RoPE phase of a slot IS its index within its own sequence,
            # which the compact plan already carries -- no need to re-derive it from a
            # cumulative count, and unused slots (comp_id == -1) get phase 0 harmlessly
            # since the masks exclude them.
            _, comp_ids, _ = self.compressor.thd_compact_plan(cu_seqlens, gstart, S_loc)
            pool_pos = comp_ids.clamp(min=0).to(device)
            cos, sin = self.rope.compress_rope(pool_pos)
            cos = cos[..., : self.rotary_dim // 2]
            sin = sin[..., : self.rotary_dim // 2]
            B = pooled.shape[0]
            cos = cos.unsqueeze(0).expand(B, -1, -1)
            sin = sin.unsqueeze(0).expand(B, -1, -1)
            pool_kv = apply_interleaved_partial_rope(
                pooled.unsqueeze(2), cos, sin, rotary_dim=self.rotary_dim
            )
            pool_kv = pool_kv.squeeze(2)
            if cp_group is not None:
                # Every query must be able to select compressed history owned by an
                # earlier rank, and the masks index the pool by GLOBAL slot, so the pool
                # has to come back to global here. Alignment guarantees each rank holds
                # the same number of windows, so the plain all-gather is well-formed and
                # its rank-major concatenation is already sequence order.
                from primus.backends.megatron.core.transformer.deepseek_v4_cp import (
                    build_global_pool,
                )

                pool_kv = build_global_pool(pool_kv, cp_group)
            return pool_kv
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
                # then drop the extra leading pool rows they produced -- nb // ratio of
                # them, not one. The two agree only while nb == ratio; overlap now asks
                # for 2 * ratio, and dropping a single row leaves a duplicate of the
                # neighbour's last window in the all-gathered pool, which shifts every
                # compressed RoPE phase and lets one key be attended to twice.
                bnd = exchange_boundary_kv(
                    hidden.reshape(hidden.shape[0], hidden.shape[1], 1, hidden.shape[2]),
                    nb,
                    cp_group,
                ).reshape(hidden.shape[0], nb, hidden.shape[2])
                pooled_local = self.compressor(torch.cat([bnd, hidden], dim=1))[
                    :, nb // self.compress_ratio :
                ]
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
        from primus.backends.megatron.core.transformer.dual_rope import mb_rope_tbl

        if mb_rope_tbl():  # membound: memoised contiguous cast table (same bytes, no per-call copies)
            cos, sin = self.rope.compress_rope.forward_arange_cast(P, device, pooled.dtype, self.rotary_dim // 2,
                                                                   stride=self.compress_ratio)
        else:
            cos, sin = self.rope.compress_rope.forward_arange(P, device, stride=self.compress_ratio)
        cos = cos[..., : self.rotary_dim // 2]
        sin = sin[..., : self.rotary_dim // 2]
        cos = cos.unsqueeze(0).expand(B, -1, -1)
        sin = sin.unsqueeze(0).expand(B, -1, -1)
        pool_kv = pooled.unsqueeze(2)  # [B, P, 1, head_dim]
        pool_kv = apply_interleaved_partial_rope(pool_kv, cos, sin, rotary_dim=self.rotary_dim)
        return pool_kv.squeeze(2)  # [B, P, head_dim]

    def _pool_or_overlap(self, hidden: torch.Tensor, cu_seqlens=None) -> torch.Tensor:
        """The compressed pool: the one forward() issued on a side stream (PRIMUS_OVERLAP_COMPRESSOR), or built now."""
        ovl, self._ovl_pool = getattr(self, "_ovl_pool", None), None
        if ovl is not None:
            br, pool = ovl
            return br.done(pool)
        return self._build_compressed_pool(hidden, cu_seqlens)

    def _hca_extra_kv(
        self,
        hidden: torch.Tensor,
        cu_seqlens=None,
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
        pool = self._pool_or_overlap(hidden, cu_seqlens)  # [B, P, head_dim]
        P = pool.shape[1]

        # Broadcast pool across all H query-heads: [B, P, head_dim] -> [B, P, H, head_dim].
        pool_h = pool.unsqueeze(2).expand(B, P, self.num_heads, self.head_dim)
        # Move heads dim to dim=1: [B, H, P, head_dim].
        pool_bh = pool_h.transpose(1, 2)

        extra_mask = self._hca_extra_mask_cached(S, P, device, dtype, cu_seqlens)
        return pool_bh, pool_bh, pool, extra_mask  # K = V = compressed pool

    def _thd_pool_mask(self, S: int, P: int, cu_seqlens, device, dtype):
        """HCA pool visibility ``[S, P]`` for packed input.

        Two conditions, both necessary: the pool slot must belong to the SAME packed
        sequence as the query, and within that sequence it must be causally visible --
        local slot ``k`` covers the sequence's rows ``[k*ratio, (k+1)*ratio)``, so it is
        visible to local query position ``u`` iff ``(k+1)*ratio - 1 <= u``. Dropping the
        same-sequence half leaks earlier samples in; dropping the causal half lets a
        token see its own future.
        """
        return _thd_pool_visibility(
            S,
            P,
            cu_seqlens,
            self.compress_ratio,
            self._thd_pool_identity(P, cu_seqlens, device),
            _v4_get_cp_group(),
            device,
            dtype,
        )

    def _thd_pool_identity(self, P: int, cu_seqlens, device):
        """``(seq_ids, comp_ids)`` for each of the P pool slots, ``-1`` where unused.

        In the compact layout the plan already carries this, so it is read from there
        rather than re-derived from a cumulative count: the masks and the pool must agree
        on what slot j IS, and deriving it twice is how they drift apart.
        """
        cp_group = _v4_get_cp_group()
        S_loc = P if cp_group is None else None  # unused; kept explicit below
        del S_loc
        parts_seq, parts_comp = [], []
        cp_size = 1 if cp_group is None else cp_group.size()
        # Every rank contributes the same c_cap slots, concatenated in rank order.
        l_local = int(cu_seqlens[-1].item()) // cp_size
        for r in range(cp_size):
            _, comp_ids, seq_ids = self.compressor.thd_compact_plan(cu_seqlens, r * l_local, l_local)
            parts_comp.append(comp_ids.to(device))
            parts_seq.append(seq_ids.to(device))
        return torch.cat(parts_seq), torch.cat(parts_comp)

    def _hca_extra_mask_cached(self, S: int, P: int, device, dtype, cu_seqlens=None):
        """HCA additive causal mask ``[S, P]``, cached (data-independent).

        Pool slot ``s`` is visible to query ``t`` iff ``(s+1)*ratio - 1 <= t``;
        the mask depends only on ``(S, P, compress_ratio, dtype)`` — all fixed
        per run — so build it once instead of rebuilding arange + where every
        compressed-layer forward. Bit-identical. PRIMUS_COMPRESS_MASK_CACHE=0
        forces the eager rebuild.
        """
        if cu_seqlens is not None:
            # Packed: visibility is per (query sequence, pool sequence) and cannot be
            # cached on (S, P) alone, since it depends on the pack's cu_seqlens.
            return self._thd_pool_mask(S, P, cu_seqlens, device, dtype)
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
        cu_seqlens=None,  # THD packed-sequence boundaries; None for BSHD
        seq_starts=None,  # [S] per-row sequence origin derived from cu_seqlens
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
        elif kv is not None and self._csa_backend == "triton_v2":
            # Without CP too, hand the triton_v2 adapter the un-broadcast [B, S, 1, D]
            # latent: its gradient then lands on `kv` directly instead of going through a
            # zeroed [B, H, S, D] buffer (268 MB memset at S=4096) and the H-fold
            # ExpandBackward sum (exactly dlatent + 63 zeros, so bit-identical).
            kv_latent = kv

        # 1) Compressed pool with compress-base RoPE.
        pool = self._pool_or_overlap(hidden, cu_seqlens)  # [B, P, head_dim]
        P = pool.shape[1]

        # 2) Indexer top-K per query. The indexer is fed a *detached* hidden
        # state: it is a selector, so the only thing that should learn from its
        # distillation loss is the indexer itself. Detaching here keeps the KL
        # from leaking into the layers below (the open-source reference detaches
        # the same way), and when the indexer is frozen it also stops autograd
        # from building a subgraph whose output is discarded.
        ovl_idx, self._ovl_indexer = getattr(self, "_ovl_indexer", None), None
        if ovl_idx is not None:  # PRIMUS_OVERLAP_INDEXER: issued on a side stream at the top of forward()
            br, out = ovl_idx
            topk_idxs, topk_scores = br.done(*out)
        else:
            topk_idxs, topk_scores = self.indexer(
                hidden.detach(), position_ids, cu_seqlens=cu_seqlens
            )  # [B, S, K]

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
                seq_starts=seq_starts,
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
        thd_local = None
        if cu_seqlens is not None:
            keys = torch.arange(q_bh.shape[2], device=q_bh.device).unsqueeze(0)
            thd_local = torch.where(
                keys >= seq_starts.to(q_bh.device, torch.long).unsqueeze(1), 0.0, float("-inf")
            ).to(q_bh.dtype)
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
            local_mask_extra=thd_local,
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
        seq_starts=None,
    ):
        """Dense (cr=0) / HCA (cr=128) dispatch on ``use_v4_attention_backend``.

        ``seq_starts`` is the THD per-row sequence origin. Only the eager path honours it
        today; the fused backends take a scalar origin and are rejected upstream in
        :meth:`forward` rather than silently ignoring it.
        """
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
                seq_starts=seq_starts,
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
        local_mask = self._local_mask(S, device=device, dtype=dtype, seq_starts=seq_starts)
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

    @staticmethod
    def _thd_local_cu(cu, global_start: int, S: int):
        """Global cu_seqlens -> this rank's own, in LOCAL row coordinates.

        Two coordinate systems are unavoidable under THD + CP: the masks compare query
        rows against POOL slots that were all-gathered back to global, so they need global
        sequence identities; but the compressor only sees this rank's ``S`` rows, so its
        window plan must be expressed locally. Feeding it the global cu_seqlens makes
        ``thd_window_plan`` emit row indices for the whole pack, which then index a tensor
        that only has ``S`` rows -- an out-of-bounds gather, which on ROCm surfaces as a
        bare HSA queue abort rather than an index error.

        A sequence straddling the shard edge is clipped, and the clipped part is a whole
        number of windows because the packer aligns every boundary to the compress ratio.
        """
        lo, hi = global_start, global_start + S
        inner = cu[(cu > lo) & (cu < hi)] - lo
        return torch.cat(
            [
                torch.zeros(1, dtype=inner.dtype, device=inner.device),
                inner,
                torch.full((1,), S, dtype=inner.dtype, device=inner.device),
            ]
        )

    def _thd_seq_starts(self, packed_seq_params, B: int, S: int, device):
        """Per-row sequence start for THD (packed) input, or ``None`` for BSHD.

        Returns an ``[S]`` int32 tensor whose element ``t`` is the first row index of the
        packed sequence that row ``t`` belongs to. Every causal / sliding-window test in
        this module is of the form "position >= 0", i.e. it compares against a SCALAR
        origin -- the start of the one sequence in the batch. Under packing there are many
        sequences in one flat row axis, so that origin becomes per-row, and this tensor is
        what turns the scalar tests into per-row ones. A token must never see across its
        own sequence's start, or the pack leaks one sample into another.

        Packing requires ``B == 1``: the pack IS the batch, cu_seqlens indexes the flat
        token axis, and a second batch dimension would need a second offset everywhere.
        """
        if packed_seq_params is None:
            return None
        cu = getattr(packed_seq_params, "cu_seqlens_q", None)
        if cu is None:
            return None
        if B != 1:
            raise RuntimeError(
                f"DeepSeek-V4 packed (THD) attention requires micro_batch_size=1 -- the pack "
                f"is the batch and cu_seqlens indexes a flat token axis; got B={B}."
            )
        cu = cu.to(device=device, dtype=torch.int64)
        # cu_seqlens always describes the WHOLE pack, even under CP where this rank holds
        # only S of its rows starting at `global_start`. Keeping it global (rather than
        # re-slicing it per rank) is deliberate: the compressed pool is all-gathered back
        # to global, so the pool-side masks need global sequence identities anyway, and
        # having one authoritative cu_seqlens avoids the two coordinate systems drifting.
        cp_group = _v4_get_cp_group()
        global_start = 0 if cp_group is None else cp_group.rank() * S
        total = S if cp_group is None else S * cp_group.size()
        if int(cu[-1].item()) != total:
            raise RuntimeError(
                f"cu_seqlens must cover the whole packed row axis: cu_seqlens[-1]="
                f"{int(cu[-1].item())} but the pack is {total} rows "
                f"(S={S}, cp_size={1 if cp_group is None else cp_group.size()}). A short "
                f"cu_seqlens silently leaves rows attending across sequence boundaries."
            )
        lengths = (cu[1:] - cu[:-1]).to(torch.int64)
        starts_global = torch.repeat_interleave(cu[:-1], lengths)  # [total]
        # This rank's rows, in LOCAL coordinates. A sequence that began before this shard
        # keeps a NEGATIVE start, deliberately: the window-validity test is
        # `candidate_row >= seq_start`, and those candidates are negative too -- they
        # index the boundary buffer holding the previous rank's rows. Clamping to 0 would
        # not leak (a straddling sequence occupies the shard's leading rows, so local
        # [0, query] is all its own), it would TRUNCATE: the sequence would lose the part
        # of its own history that lives on the neighbour.
        local = starts_global[global_start : global_start + S] - global_start
        return local.to(torch.int32)

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """``[B, S, D] -> [B, S, D]``.

        Dispatches on ``self.compress_ratio``:

        * ``0``   — dense / SWA over local KV (single key axis).
        * ``128`` — HCA: concat compressed pool to local KV, joint softmax.
        * ``4``   — CSA: per-query top-K from compressed pool, joint softmax.

        ``packed_seq_params`` carries THD (packed-sequence) cu_seqlens. When it is
        ``None`` -- the ordinary BSHD path -- every branch below behaves exactly as
        before; ``_thd_seq_starts`` returns ``None`` and each index construction falls
        back to its scalar-origin form.
        """
        B, S, _ = hidden.shape
        seq_starts = self._thd_seq_starts(packed_seq_params, B, S, hidden.device)
        # The compressed branches need the boundaries themselves, not just the per-row
        # origin: pooling, pool RoPE phase and pool visibility are all per sequence.
        cu_seqlens = (
            None
            if seq_starts is None
            else packed_seq_params.cu_seqlens_q.to(device=hidden.device, dtype=torch.int64)
        )
        if seq_starts is not None and self._attn_backend not in ("eager", "triton_v2"):
            raise NotImplementedError(
                "DeepSeek-V4 packed (THD) attention is implemented for the eager and "
                f"triton_v2 backends; got '{self._attn_backend}'. The others build their "
                "index matrix from a SCALAR sequence origin, so running them under packing "
                "would silently let each sample attend to its predecessors rather than "
                "fail."
            )
        device, dtype = hidden.device, hidden.dtype

        # PRIMUS_OVERLAP_INDEXER=1: the CSA indexer (forward only: detached input, frozen params, integer top-k out)
        # depends on nothing but `hidden`, so it is issued on a side stream here and runs next to the q / kv
        # projections and the compressor; _csa_forward waits for it. Same kernels and inputs -> same top-k.
        self._ovl_indexer = None
        if (
            self.compress_ratio == 4
            and self.indexer is not None
            and _overlap is not None
            and _overlap.enabled("INDEXER")
            and cu_seqlens is None
            and _v4_get_cp_group() is None
            and not (self.indexer_distill_enabled and self.training and torch.is_grad_enabled())
            and not any(p.requires_grad for p in self.indexer.parameters())
        ):
            with _overlap.branch("indexer") as br:
                out = self.indexer(hidden.detach(), position_ids, cu_seqlens=None)
            self._ovl_indexer = (br, out)

        # PRIMUS_OVERLAP_COMPRESSOR (HCA / CSA): the compressor (pool) depends only on `hidden`, so it is issued on a side
        # stream next to the q / kv path (its backward follows it there). ForkGrad sums the two input gradients in the
        # serial engine's order -- compressor first, then the qkv_a projection (the only other consumer on the concat
        # path) -- so every bit stays the same. =1: HCA layers only; =2: CSA layers too (next to the indexer branch).
        self._ovl_pool = None
        if (
            self.compressor is not None
            and _overlap is not None
            and _overlap.enabled("COMPRESSOR")
            and (self.compress_ratio == 128 or (self.compress_ratio == 4 and _overlap.level("COMPRESSOR") >= 2))
            and cu_seqlens is None
            and _v4_get_cp_group() is None
            and hidden.requires_grad
            and torch.is_grad_enabled()
            and self._qkv_a_concat_ok(hidden)
            and not (self.indexer_distill_enabled and self.training)
        ):
            h_q, h_c = _overlap.ForkGrad.apply(hidden, 2, (1, 0))
            with _overlap.branch("compressor") as brc:
                pool_ovl = self._build_compressed_pool(h_c, None)
            self._ovl_pool = (brc, pool_ovl)
            hidden = h_q

        # PRIMUS_V4_QKV_A_CONCAT=1: wq_a and wkv as one GEMM (see _QKVASplit); else two linears.
        q_a, kv_a = None, None
        if self._qkv_a_concat_ok(hidden):
            slot = self._fuse2_att_gadd_slot(hidden, seq_starts)
            if slot is not None:
                # PRIMUS_FUSE2_ATT_GADD: the concat and the compressor get their own aliases of the input; the
                # concat's dX GEMM adds into the compressor's dX (autograd's bf16 add, in the GEMM epilogue)
                from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.fuse2_main_grad import (
                    AttFanOut,
                )
                from primus_turbo.pytorch.ops.gemm_fp8_concat import request_dx_accum

                h_cat, hidden = AttFanOut.apply(hidden, slot)
                request_dx_accum(slot)
                try:
                    q_a, kv_a = self._qkv_a_concat(h_cat)
                finally:
                    request_dx_accum(None)
                self.compressor._fuse2_dx_slot = slot
            else:
                q_a, kv_a = self._qkv_a_concat(hidden)
        if (
            q_a is not None
            and self._fused_q_norm_rope_ok(hidden)
            and qkv_post_supported(q_a, kv_a, self.q_layernorm, self.kv_layernorm, self.rotary_dim)
        ):
            # PRIMUS_V4_QKV_POST_FUSED=1: q_layernorm, kv_layernorm and the kv RoPE as one op (fwd 3 -> 1 launch,
            # bwd 5 -> 2), bit-identical (see qkv_post.py).
            rope = self.rope.get_rope(compress_ratio=self.compress_ratio)
            q_c, kv_r = qkv_post(
                q_a, kv_a, self.q_layernorm.weight, self.kv_layernorm.weight,
                position_ids.broadcast_to((B, S)).reshape(-1), rope.inv_freq,
                eps_q=self.q_layernorm.eps, eps_kv=self.kv_layernorm.eps, rotary_dim=self.rotary_dim,
                q_first=getattr(self, "_qkv_a_fp8_order", "q_kv") == "q_kv",
            )
            q = self._apply_q_norm_rope(hidden, position_ids, q_compressed=q_c, q_normed=True)
            kv = kv_r.view(B, S, 1, self.head_dim)
        elif self._fused_q_norm_rope_ok(hidden):
            # One pass for per-head RMSNorm + RoPE on Q, written straight into the
            # zero-padded buffer the sparse-MLA adapter consumes (no pad copy).
            q = self._apply_q_norm_rope(hidden, position_ids, q_compressed=q_a)  # [B, S, H, head_dim] view
            kv = self._apply_kv(hidden, kv=kv_a)  # [B, S, 1, head_dim]
            kv = self.rope.apply_rope(kv, position_ids=position_ids, compress_ratio=self.compress_ratio)
        else:
            q = self._apply_q(hidden, q_compressed=q_a)  # [B, S, H, head_dim]
            kv = self._apply_kv(hidden, kv=kv_a)  # [B, S, 1, head_dim]

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
                seq_starts=seq_starts,
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
            _, _, pool, extra_mask = self._hca_extra_kv(hidden, cu_seqlens)
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
                seq_starts=seq_starts,
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
            out_bh = self._csa_forward(
                hidden, q_bh, k_local_bh, v_local_bh, None, position_ids, kv, cu_seqlens, seq_starts
            )
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
