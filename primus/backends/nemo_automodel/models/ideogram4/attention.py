###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Var-len (``cu_seqlens``) flash-attention processor for Ideogram-4 in the NeMo
AutoModel diffusion recipe — the fast, exact replacement for masked torch SDPA.

WHY:
  Ideogram-4's attention runs on **masked torch SDPA** today: the transformer builds
  a dense ``(B,1,L,L)`` block-diagonal boolean mask from ``segment_ids`` and hands it
  to ``dispatch_attention_fn``. A dense mask forces the SDPA/NATIVE backend (flash /
  aiter backends reject one), so the whole model forgoes flash — and at 1k–2k px
  resolutions attention is O(L²) and dominates the step. The mask is, in practice,
  a per-row block-diagonal over CONTIGUOUS segments (left-padding = {pad}+{text+image};
  future multi-sample packing = several sample blocks). That maps exactly onto
  **variable-length flash attention** (``cu_seqlens``): pack the batch into one flat
  sequence and mark segment boundaries — no dense mask, full flash speed, EXACT numerics.

WHAT (NO diffusers / Automodel fork):
  A drop-in diffusers attention processor for ``Ideogram4Attention``. It reproduces the
  stock ``Ideogram4AttnProcessor`` math verbatim (q/k/v proj → q/k RMSNorm → MRoPE) and
  only replaces the attention CALL: it converts the block-diagonal boolean mask to
  ``cu_seqlens`` and runs ``aiter.flash_attn_varlen_func`` (``deterministic=False`` — the
  non-deterministic backward is a large, numerically-equivalent throughput win, and the
  *deterministic* hd=256 backward has a workspace that OOMs at 2048²). When the mask is
  absent or trivial (one full segment per row, e.g. a fixed-length/no-pad batch) it takes
  the plain dense ``flash_attn_func`` fast path.

  Install swaps the class default processor (``Ideogram4Attention._default_processor_cls``)
  BEFORE the recipe builds the model, so every attention module is constructed with it.
  Env-gated by ``PRIMUS_IDEOGRAM_VARLEN_ATTN=1`` (default off = stock SDPA path). No
  Automodel/diffusers source is modified.

GENERALITY / REUSE:
  The mask→``cu_seqlens`` transform (:func:`blockdiag_bool_mask_to_cu_seqlens`) and the
  var-len call (:func:`varlen_flash_attention`) are **model-agnostic** — they assume only
  a per-row block-diagonal boolean mask with contiguous segments, so any packed/padded
  attention (not just Ideogram) can reuse them. The only Ideogram-specific piece is the
  thin processor that wires the model's proj/norm/RoPE to them.

CORRECTNESS:
  Exact, not approximate. Every token (including padding) keeps its own segment, so the
  result matches masked SDPA on all positions up to bf16 + non-deterministic-atomic
  ordering (image-token velocity — the only positions the loss reads — matches within the
  bf16 floor). The processor falls back to the ORIGINAL dense dispatch ONLY for mask
  *types* it cannot represent as contiguous segments (a non-boolean/additive mask), and
  warns once — it never silently relays the unoptimized path for the case it exists to
  serve (Ideogram always passes a boolean block-diagonal mask).

COMPILE:
  The mask->``cu_seqlens`` transform is data-dependent (``.any()``, ``nonzero``, ``.item()``),
  so it forces device->host reads that ``torch.compile`` cannot trace -- each is a graph break,
  and under FSDP2 a mid-forward break splits the region the per-layer all-gather/reshard
  collectives are registered around and desyncs them across ranks (a silent multi-rank death).
  The fix is to not derive the packing here at all: ``Ideogram4Adapter`` builds ``cu_seqlens``
  on the host (it already holds the text lengths as Python ints) and publishes it into a shared
  non-persistent buffer on the attention modules, making it a graph INPUT (see
  ``packing_buffer.py`` for why that route and not ``attention_kwargs``, which is a
  dead parameter for this model in diffusers 0.39.0). That path is exact on ragged batches,
  needs no host sync, and stops recomputing the same packing once per layer.

KERNEL FAMILY:
  The var-len call goes to aiter's **Triton** kernels under Primus's own head-dim-256 block
  sizes (``triton_varlen_attn.py``): 2.1x faster than CK-tile on the production packing,
  exact to CK element-wise, and worth -20% end-to-end step time at 8 ranks / mbs=8 / 1024px
  / compile (3.125 s -> 2.499 s on ZeRO-2, 3.274 -> 2.625 on ZeRO-3, same peak memory and
  loss). ``model.varlen_attn_impl: ck`` in the model preset returns to the CK-tile kernels.
  The choice covers the var-len route only -- the dense fallbacks stay on CK, the shape they
  were measured on. On a GPU or head dim nobody swept, ``triton`` degrades to ``ck`` at
  install time (see :func:`resolve_varlen_impl`), because there the Triton path would run
  aiter's head-dim-blind config and LOSE to CK.

Activation (env, no config schema change):
    PRIMUS_IDEOGRAM_VARLEN_ATTN=1          swap Ideogram-4 attention to the var-len flash path
    PRIMUS_IDEOGRAM_PRECOMPUTE_CU_SEQLENS  default 1; set 0 to stop the adapter precomputing
                                           the packing and fall back to the mask-derived
                                           (host-syncing) path. Escape hatch for A/B and
                                           rollback.
    PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE=1    skip the mask->cu_seqlens host-sync and use dense
                                           flash directly (torch.compile-safe; EXACT only for
                                           equal-length / unpadded batches, e.g. fixed-text).
                                           Never fires while a precomputed cu_seqlens is
                                           provided, which now takes precedence.
    PRIMUS_IDEOGRAM_VARLEN_ATTN_IMPL       override model.varlen_attn_impl (ck | triton) for
                                           an A/B run without editing the preset.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
from torch import Tensor

from primus.backends.nemo_automodel.models.ideogram4.packing_buffer import (
    resolve_packing,
)

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "True", "yes", "on"}


def is_varlen_attn_enabled() -> bool:
    """Whether the Ideogram-4 var-len flash-attention processor should be installed."""
    return os.getenv("PRIMUS_IDEOGRAM_VARLEN_ATTN", "0") in _TRUTHY


def assume_dense_enabled() -> bool:
    """Whether to skip the block-diagonal mask analysis and use dense flash directly.

    The mask->``cu_seqlens`` transform does data-dependent host-syncs (``bool(.any())``,
    ``.max().item()``, ``nonzero``) that graph-break ``torch.compile`` and, under FSDP2
    multi-rank, desync the per-layer collectives. When every row is a single full segment
    (an equal-length / unpadded batch -> the mask is trivial anyway), set
    ``PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE=1`` to skip the analysis and go straight to dense
    flash, keeping the compiled per-layer graph break-free. EXACT only for equal-length /
    no-pad batches (e.g. fixed-text); do NOT set it for ragged/padded batches.

    Superseded in practice by :func:`precompute_cu_seqlens_enabled` -- when the adapter
    hands the packing in, the processor prefers it and this flag never fires.
    """
    return os.getenv("PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE", "0") in _TRUTHY


def precompute_cu_seqlens_enabled() -> bool:
    """Whether the adapter precomputes ``cu_seqlens`` on the host. Default ON.

    The adapter already holds the per-sample text lengths as Python ints, so it can build
    the var-len packing itself and pass it into the processor as a plain tensor -- a graph
    INPUT rather than something derived from the mask mid-graph. That removes the
    device->host syncs, so the exact var-len path compiles on ragged batches.

    Set ``PRIMUS_IDEOGRAM_PRECOMPUTE_CU_SEQLENS=0`` to restore the previous behaviour
    (mask-derived packing, or dense flash under ``ASSUME_DENSE``). Kept as an escape hatch
    for A/B measurement and rollback, not as a routine knob.
    """
    return os.getenv("PRIMUS_IDEOGRAM_PRECOMPUTE_CU_SEQLENS", "1") in _TRUTHY


_VALID_IMPLS = ("ck", "triton", "asm")


def varlen_attn_impl() -> str:
    """Which kernel family the var-len path asks for: ``triton`` (default) or ``ck``.

    Selected by ``model.varlen_attn_impl`` in the Ideogram-4 model preset, because this is
    a per-model kernel choice that belongs with the model config rather than in the
    environment. ``PRIMUS_IDEOGRAM_VARLEN_ATTN_IMPL`` overrides it for A/B runs without
    editing the YAML.

    This is the REQUEST; :func:`resolve_varlen_impl` is what the run actually gets.
    """
    from primus.backends.nemo_automodel.argument_builder import get_param

    impl = os.getenv("PRIMUS_IDEOGRAM_VARLEN_ATTN_IMPL") or get_param("model.varlen_attn_impl", "triton")
    impl = str(impl).strip().lower()
    if impl not in _VALID_IMPLS:
        raise ValueError(
            f"model.varlen_attn_impl={impl!r} is not one of {_VALID_IMPLS}. "
            "Set it in the Ideogram-4 model preset (or PRIMUS_IDEOGRAM_VARLEN_ATTN_IMPL)."
        )
    return impl


def resolve_varlen_impl() -> str:
    """The kernel family the run will really use, after the tuned-shape guard.

    ``triton`` only beats CK where Primus has measured block sizes for this
    ``(arch, head_dim)``. Everywhere else the Triton kernels fall back to aiter's
    head-dim-blind config, which at hd=256 is 2.0x SLOWER than CK -- so an unswept GPU
    degrades to ``ck`` rather than to a silent regression. That is what makes ``triton``
    defensible as the shipped default instead of an opt-in.

    Resolved ONCE, by :func:`install`, and stored on the processor: the answer is fixed for
    the process, and asking per call would put a branch and a logging side effect inside the
    compiled graph -- exactly what this file's COMPILE section exists to avoid.
    """
    impl = varlen_attn_impl()

    if impl == "asm":
        # Hand-written CDNA4 assembly for the forward AND the backward. It is built for
        # gfx950 at head_dim 256 specifically -- the tile sizes were tuned on that shape --
        # and it lives outside the Primus tree, so anything missing degrades to the shipped
        # default rather than failing the run.
        try:
            from primus.backends.nemo_automodel.models.ideogram4.asm_varlen_attn_shim import (
                asm_available,
            )

            ok, why = asm_available()
        except Exception as exc:  # pragma: no cover - import-time environment issue
            ok, why = False, f"{type(exc).__name__}: {exc}"
        if not ok:
            _warn_once(
                "unavailable_asm",
                f"[PrimusIdeogramVarlen] varlen_attn_impl=asm is unavailable ({why}). "
                "Falling back to impl=triton.",
            )
            return "triton"
        return impl

    if impl != "triton":
        return impl

    from primus.backends.nemo_automodel.models.ideogram4.triton_varlen_attn import (
        IDEOGRAM4_HEAD_DIM,
        is_tuned,
    )

    if not is_tuned(IDEOGRAM4_HEAD_DIM):
        _warn_once(
            "untuned_triton",
            "[PrimusIdeogramVarlen] varlen_attn_impl=triton, but this GPU has no tuned "
            f"head-dim-{IDEOGRAM4_HEAD_DIM} block sizes in triton_varlen_attn._TUNED_DELTAS; "
            "aiter's head-dim-blind config would be slower than CK-tile here. Falling back "
            "to impl=ck. Sweep this arch (scripts/bench_hd256_varlen_attn.py) and add its "
            "row to enable Triton.",
        )
        return "ck"
    return impl


def precompute_cu_seqlens_active() -> bool:
    """Whether precomputing the packing will actually be *read* by anything.

    Both switches have to be on. Without ``PRIMUS_IDEOGRAM_VARLEN_ATTN`` the stock SDPA
    processor is in place and has no ``cu_seqlens`` parameter, so precomputing would cost a
    build plus the reserved pad column's token position every step for nothing. This is the
    gate the adapter uses; :func:`precompute_cu_seqlens_enabled` is only the flag.
    """
    return is_varlen_attn_enabled() and precompute_cu_seqlens_enabled()


# --------------------------------------------------------------------------- #
# Model-agnostic helpers (reusable for any packed/padded block-diagonal attn)  #
# --------------------------------------------------------------------------- #
def blockdiag_bool_mask_to_cu_seqlens(mask: Tensor) -> Tuple[Tensor, int, bool]:
    """Convert a block-diagonal boolean attention mask to a var-len packing.

    Args:
        mask: ``(B, 1, L, L)`` or ``(B, L, L)`` boolean tensor, ``True`` = "query i
            attends to key j". Each row ``b`` must be block-diagonal over CONTIGUOUS
            segments (a segment boundary lies between positions ``i`` and ``i+1``
            wherever ``mask[b, .., i, i+1]`` is ``False``); segments never span the row
            boundary. This is what an ``(seg_i == seg_j)`` mask produces for
            contiguously-assigned ``segment_ids`` (padding, or packed samples).

    Returns:
        ``(cu_seqlens, max_seqlen, is_trivial)`` describing the packed layout over the
        flattened ``(B*L)`` sequence:
          * ``cu_seqlens``: ``int32`` ``(num_segments + 1,)`` cumulative segment lengths,
            starting at 0 and ending at ``B*L`` (the ``flash_attn_varlen_func`` contract).
          * ``max_seqlen``: longest segment length.
          * ``is_trivial``: ``True`` iff every row is a single full segment (no internal
            splits) — the caller may then use plain dense attention and skip packing.
    """
    if mask.dim() == 4:
        m = mask[:, 0]
    elif mask.dim() == 3:
        m = mask
    else:
        raise ValueError(f"expected a (B,1,L,L) or (B,L,L) mask, got shape {tuple(mask.shape)}")
    if m.dtype != torch.bool:
        raise TypeError(f"expected a boolean mask, got dtype {m.dtype}")

    B, L, L2 = m.shape
    if L != L2:
        raise ValueError(f"mask must be square in the last two dims, got {(L, L2)}")
    device = m.device

    if L == 1:
        cu = torch.arange(0, B + 1, dtype=torch.int32, device=device)
        return cu, 1, True

    # A new segment starts at position i+1 within a row wherever i and i+1 do not attend
    # (the sub-diagonal test is all we need for contiguous block-diagonal segments).
    superdiag = m.diagonal(offset=1, dim1=-2, dim2=-1)  # (B, L-1) : mask[b, i, i+1]
    splits = ~superdiag  # (B, L-1)
    is_trivial = not bool(splits.any())

    # Per-token "starts a new segment" over the flattened (B, L) sequence. The first
    # token of every row always starts a segment (segments never cross rows).
    new_seg = torch.zeros(B, L, dtype=torch.bool, device=device)
    new_seg[:, 0] = True
    new_seg[:, 1:] = splits
    seg_starts = torch.nonzero(new_seg.reshape(-1), as_tuple=False).flatten()

    total = B * L
    cu = torch.empty(seg_starts.numel() + 1, dtype=torch.int32, device=device)
    cu[:-1] = seg_starts.to(torch.int32)
    cu[-1] = total
    max_seqlen = int((cu[1:] - cu[:-1]).max().item())
    return cu, max_seqlen, is_trivial


def _unwrap(out):
    """aiter returns (out, lse, ...) when return_lse=True; keep only the output."""
    return out[0] if isinstance(out, (tuple, list)) else out


def dense_flash_attention(q: Tensor, k: Tensor, v: Tensor, *, deterministic: bool = False) -> Tensor:
    """Plain (unmasked) bf16 flash attention. q/k/v: ``(B, L, H, D)`` -> ``(B, L, H, D)``.

    ``return_lse=True`` is required by aiter's autograd forward (LSE is saved for the
    backward); we request it and drop the LSE.
    """
    import aiter

    return _unwrap(aiter.flash_attn_func(q, k, v, causal=False, deterministic=deterministic, return_lse=True))


def varlen_flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    *,
    deterministic: bool = False,
    impl: str = "ck",
) -> Tensor:
    """Variable-length bf16 flash attention over a packed sequence.

    q/k/v: ``(total_tokens, H, D)`` (packed, no padding between segments). ``cu_seqlens``
    and ``max_seqlen`` come from :func:`blockdiag_bool_mask_to_cu_seqlens`. Returns
    ``(total_tokens, H, D)``.

    ``impl`` picks the kernel family (see :func:`varlen_attn_impl`). ``deterministic`` is
    read only by ``ck``; the Triton one-kernel backward computes dq/dk/dv in separate
    passes with no atomics, so it is deterministic either way.
    """
    if impl == "triton":
        from primus.backends.nemo_automodel.models.ideogram4.triton_varlen_attn import (
            triton_varlen_flash_attention,
        )

        return triton_varlen_flash_attention(q, k, v, cu_seqlens, max_seqlen)

    if impl == "asm":
        from primus.backends.nemo_automodel.models.ideogram4.asm_varlen_attn_shim import (
            asm_varlen_flash_attention,
        )

        return asm_varlen_flash_attention(q, k, v, cu_seqlens, max_seqlen)

    import aiter

    return _unwrap(
        aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=False,
            deterministic=deterministic,
            return_lse=True,
        )
    )


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate-half, matching diffusers ``transformer_ideogram4._rotate_half`` exactly."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# --------------------------------------------------------------------------- #
# Ideogram-4 var-len attention processor                                       #
# --------------------------------------------------------------------------- #
_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg)


class Ideogram4VarlenAttnProcessor:
    """Ideogram-4 self-attention via var-len flash (exact block-diagonal packing).

    Mirrors diffusers ``Ideogram4AttnProcessor`` (q/k/v proj, q/k RMSNorm, MRoPE, output
    proj) and swaps only the attention call for a ``cu_seqlens`` flash path. Non-det
    backward (``deterministic=False``).

    ``_attention`` picks a path in this order:

      1. **provided** ``cu_seqlens`` -- the adapter precomputed the packing on the host and
         published it into the shared buffer this processor reads off ``attn``. Exact on
         ragged batches and free of data-dependent ops, so it is the only path that is
         simultaneously correct on ragged data and safe under per-layer ``torch.compile``
         + FSDP2.
      2. ``assume_dense`` **or no mask** -- dense flash. Exact only when no row has padding.
      3. **mask analysis** (legacy) -- exact, but its device->host reads graph-break.
    """

    # kept for API-compatibility with diffusers' processor discovery / set_attention_backend
    _attention_backend = None
    _parallel_config = None

    deterministic: bool = False
    # Read once at class-definition time (torchrun sets env before import). Keeps the check a
    # constant attribute lookup inside the compiled graph (no data-dependent branch / break).
    assume_dense: bool = assume_dense_enabled()
    # Kernel family for the var-len path; the preset's default is ``triton``. Resolved by
    # ``install()`` rather than here, because the YAML is only published once the trainer has
    # merged it (still before the model is built) and because the tuned-shape guard needs a
    # GPU to inspect. This class default is only what a processor built WITHOUT install()
    # gets, so it stays the conservative kernel. Either way it is a constant attribute
    # lookup inside the graph, not a branch on run state.
    varlen_impl: str = "ck"

    def __call__(
        self,
        attn,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        image_rotary_emb: Tuple[Tensor, Tensor],
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tensor:
        # The packing normally arrives on the module: the adapter publishes it into a shared
        # non-persistent buffer that ``resolve_packing`` reads off ``attn`` (see
        # ``packing_buffer.py``). The two named parameters are kept because
        # diffusers' attention module filters forwarded kwargs against
        # ``inspect.signature(self.processor.__call__).parameters``, so declaring them by name
        # is what would let a kwargs route work at all -- a ``**kwargs``-only processor would
        # silently receive nothing. An explicit argument wins over the buffer.
        cu_seqlens, max_seqlen = resolve_packing(attn, cu_seqlens, max_seqlen)

        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        cos, sin = image_rotary_emb
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        query = (query * cos) + (_rotate_half(query) * sin)
        key = (key * cos) + (_rotate_half(key) * sin)

        out = self._attention(query, key, value, attention_mask, cu_seqlens, max_seqlen)
        out = out.flatten(2, 3)
        return attn.to_out[0](out)

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tensor:
        B, L, H, D = query.shape

        # No mask, or a mask type we cannot represent as contiguous segments: for a
        # boolean block-diagonal mask we always take the exact var-len path (that is the
        # case this processor exists for); a non-boolean/additive mask is not something
        # Ideogram emits, so defer to the original dense dispatch rather than guess.
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            _warn_once(
                "nonbool_mask",
                "[PrimusIdeogramVarlen] non-boolean attn_mask -> dense SDPA dispatch fallback.",
            )
            from diffusers.models.attention_dispatch import dispatch_attention_fn

            return dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )

        # MRoPE multiplies q/k by the float32 cos/sin, promoting them to float32. Torch
        # SDPA is autocast-aware and downcasts at its boundary; aiter's flash op is not,
        # so match that here by casting q/k to the projection (compute) dtype. ``value``
        # skips RoPE and already carries it.
        compute_dtype = value.dtype
        query = query.to(compute_dtype)
        key = key.to(compute_dtype)

        # PREFERRED PATH: the packing was precomputed on the host and handed in, so nothing
        # here inspects tensor values. Exact on ragged batches AND compile-safe -- this is
        # the only combination of those two properties available (aiter.flash_attn_varlen_func
        # is itself traceable, so the whole processor stays in one graph and the FSDP2
        # per-layer collectives keep their ordering). ``attention_mask`` is deliberately not
        # read on this path; the model still materializes it, which is now dead weight.
        # The branch tests a property of the RUN (was anything provided), not of the data,
        # so it costs one guard and no graph break.
        if cu_seqlens is not None:
            # The packing lives on the module and outlives the step that published it, so a
            # caller that bypasses the adapter (a sampling pass, an eval loop with a different
            # batch size) could otherwise attend on a stale one and corrupt silently. This
            # compares only static SHAPE metadata -- no values are read, so it costs a guard
            # and no host sync. ``2B+1`` is the layout contract: one pad segment plus one
            # text+image segment per row. Packing several samples per row would change that
            # count, and this check with it.
            expected = 2 * B + 1
            if cu_seqlens.numel() != expected:
                raise ValueError(
                    f"cu_seqlens has {cu_seqlens.numel()} entries but this batch needs "
                    f"{expected} (2*B+1 for B={B}). Either the packing was published for a "
                    "different batch size and is stale -- call "
                    "ideogram4_packing_buffer.clear_packing(model) before running the model "
                    "outside the adapter -- or the sequence layout no longer has exactly two "
                    "segments per row, in which case update this check."
                )
            # Hand the kernel a PRIVATE COPY, never the shared buffer. aiter's var-len op
            # treats cu_seqlens as a mutable argument: it saves the tensor for its backward
            # and then writes it, bumping the autograd version counter once per call. One
            # buffer shared by 34 layers therefore has its version moved 34 times per forward
            # while each layer's backward still expects the version it saved, and the step
            # dies in .backward() with "a variable needed for gradient computation has been
            # modified by an inplace operation: IntTensor[5] is at version 35; expected 34"
            # (measured 2026-08-04, t2 stage; the legacy path never saw it because every
            # layer derives its own tensor from the mask). The clone is 5 int32 already on
            # device, and under compile it is a graph intermediate -- the buffer stays a
            # read-only graph input, which is what makes sharing it safe at all.
            out = varlen_flash_attention(
                query.reshape(B * L, H, D),
                key.reshape(B * L, H, D),
                value.reshape(B * L, H, D),
                cu_seqlens.clone(),
                L if max_seqlen is None else max_seqlen,
                deterministic=self.deterministic,
                impl=self.varlen_impl,
            )
            return out.reshape(B, L, H, D)

        # Dense fast path. ``attention_mask is None`` OR the equal-length assertion
        # (``assume_dense``) skips the data-dependent mask->cu_seqlens host-sync, so under
        # torch.compile the whole processor stays in one graph (aiter.flash_attn_func is
        # itself compile-safe -- fullgraph traces it with no break), which is required for
        # FSDP2 multi-rank compile to not desync collectives. Exact for equal-length/no-pad
        # batches; ragged/padded batches must leave PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE unset.
        if attention_mask is None or self.assume_dense:
            return dense_flash_attention(query, key, value, deterministic=self.deterministic)

        # LEGACY PATH: derive the packing from the mask. Exact, but the derivation does
        # device->host reads, so it graph-breaks and is unsafe under multi-rank compile.
        # Reached only when nothing was precomputed (kill switch, or a non-adapter caller).
        # Retained as the reference implementation the unit test checks against.
        cu_from_mask, max_from_mask, is_trivial = blockdiag_bool_mask_to_cu_seqlens(attention_mask)
        if is_trivial:
            return dense_flash_attention(query, key, value, deterministic=self.deterministic)

        q = query.reshape(B * L, H, D)
        k = key.reshape(B * L, H, D)
        v = value.reshape(B * L, H, D)
        out = varlen_flash_attention(
            q,
            k,
            v,
            cu_from_mask,
            max_from_mask,
            deterministic=self.deterministic,
            impl=self.varlen_impl,
        )
        return out.reshape(B, L, H, D)


# --------------------------------------------------------------------------- #
# Install                                                                      #
# --------------------------------------------------------------------------- #
def install(model=None) -> bool:
    """Route Ideogram-4 attention through the var-len flash processor (no-fork).

    No-op (returns False) unless ``PRIMUS_IDEOGRAM_VARLEN_ATTN`` is set. Patches the
    class default processor so every ``Ideogram4Attention`` built AFTER this call uses
    the var-len processor; if a built ``model`` is passed, also swaps its existing
    modules. Idempotent. Modifies NO Automodel/diffusers source.
    """
    if not is_varlen_attn_enabled():
        return False

    # Fail fast if aiter's flash-attention is unavailable, so the run errors clearly
    # rather than silently keeping the SDPA path.
    import aiter  # noqa: F401
    from diffusers.models.transformers.transformer_ideogram4 import Ideogram4Attention

    impl = resolve_varlen_impl()
    Ideogram4VarlenAttnProcessor.varlen_impl = impl

    already = getattr(Ideogram4Attention, "_primus_varlen_installed", False)
    if not already:
        Ideogram4Attention._default_processor_cls = Ideogram4VarlenAttnProcessor
        if Ideogram4VarlenAttnProcessor not in Ideogram4Attention._available_processors:
            Ideogram4Attention._available_processors = [
                *Ideogram4Attention._available_processors,
                Ideogram4VarlenAttnProcessor,
            ]
        Ideogram4Attention._primus_varlen_installed = True

    swapped = 0
    if model is not None:
        for module in model.modules():
            if isinstance(module, Ideogram4Attention) and not isinstance(
                module.processor, Ideogram4VarlenAttnProcessor
            ):
                module.set_processor(Ideogram4VarlenAttnProcessor())
                swapped += 1

    if already and swapped == 0:
        return True

    logger.info(
        "[PrimusIdeogramVarlen] Installed var-len flash-attention processor for "
        "Ideogram4Attention (impl=%s, deterministic=False)%s.",
        impl,
        f"; swapped {swapped} existing module(s)" if swapped else "",
    )
    if impl == "triton":
        from primus.backends.nemo_automodel.models.ideogram4.triton_varlen_attn import (
            IDEOGRAM4_HEAD_DIM,
            describe_config,
        )

        logger.info("[PrimusIdeogramVarlen] Triton %s", describe_config(IDEOGRAM4_HEAD_DIM))
    return True
