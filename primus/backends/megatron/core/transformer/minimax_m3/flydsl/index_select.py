###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The MiniMax-M3 indexer's block selection, fused: O(S * n_blocks) memory.

Forward: ``index_block_max`` scores and max-pools in one kernel, then the
indexer's own boost + top-k runs on the ``[b, n, S, n_blocks]`` block scores.
When a backward will follow, the selection's :class:`BlockPlan` -- the block
table inverted into per-block entry lists -- is built here once and handed to
the caller, whose attention backward walks the same lists.

Backward: a block score is the max over its keys, so its gradient lands on the
one winning key (the kernel's argmax) -- ``d q_t += g * k_j*`` and
``d k_j* += g * q_t``. Only the selected, non-forced slots can carry a
gradient: the sparse loss reads nothing else, and a forced block's +inf was
written over its score. So the backward gathers at those <= topk slots per
row instead of touching the whole block axis. dQ sums each token's slots
straight from the key rows (``index_dq``). dK walks the plan block by block
(``index_dk``): a block's keys only receive gradient from the slots that
picked it, which is exactly a plan row -- deterministic, no atomics, and no
sort of its own.
"""

import functools

import torch

from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from primus.backends.megatron.core.transformer.minimax_m3.flydsl import select_plan
from primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_block_max import (
    index_block_max,
    index_dk,
    index_dq,
)
from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
    build_block_plan,
)
from primus.core.utils.module_utils import warning_rank_0

_UNSUPPORTED_WARNED = False


def index_dk_scatter_enabled(config) -> bool:
    """``msa_index_dk_scatter`` as the indexer's backward uses it: the switch, unless
    ``msa_exact_numerics`` is set (relaxed numerics: the chunk partials of ``index_dk`` are
    summed in another order and from the full fp32 ``g``). Decided before any work; whether
    the kernel serves the shape and passes its first-call comparison is ``index_dk``'s part."""
    return bool(getattr(config, "msa_index_dk_scatter", False)) and not bool(
        getattr(config, "msa_exact_numerics", False)
    )


def bounded_plan_chunks_enabled(config) -> bool:
    """``msa_bounded_plan_chunks`` as the selection builds its :class:`BlockPlan`: the switch,
    unless ``msa_exact_numerics`` is set (relaxed numerics: the chunk partials of dK / dV and of
    the index dK are summed over other groups of entries). Decided before any work, from the
    config alone; the chunk size itself is a function of ``B * Hkv * S * topk``
    (``msa_token_bwd.plan_chunk_size``)."""
    return bool(getattr(config, "msa_bounded_plan_chunks", False)) and not bool(
        getattr(config, "msa_exact_numerics", False)
    )


def _fused_mode(select_from_block_scores, B, H, S, NB):
    """Which form of the fused selection serves this shape, decided from the config and the
    shape before any work: ``"plan"`` (:func:`select_plan.supported`), ``"wide"``
    (``msa_select_plan_wide`` and :func:`select_plan.wide_supported`) or None (the unfused
    selection)."""
    from primus.backends.megatron.core.transformer.minimax_m3.indexer import (
        MinimaxM3Indexer,
    )

    indexer = getattr(select_from_block_scores, "__self__", None)
    # select_plan re-implements both methods, so neither may be overridden
    if (
        not isinstance(indexer, MinimaxM3Indexer)
        or not getattr(indexer.config, "msa_fused_select_plan", False)
        or type(indexer).select_from_block_scores is not MinimaxM3Indexer.select_from_block_scores
        or type(indexer)._boost_always_visible is not MinimaxM3Indexer._boost_always_visible
    ):
        return None
    topk = min(indexer.topk_blocks, NB)
    if select_plan.supported(B, H, S, NB, topk):
        return "plan"
    wide = bool(getattr(indexer.config, "msa_select_plan_wide", False))
    if wide and select_plan.wide_supported(B, H, S, NB, topk):
        return "wide"
    global _UNSUPPORTED_WARNED
    if not _UNSUPPORTED_WARNED:
        _UNSUPPORTED_WARNED = True
        warning_rank_0(
            f"MSA indexer: fused block selection off for B={B}, H={H}, n_blocks={NB}, topk={topk} "
            "(needs n_blocks <= 64, topk <= 32, B*H*n_blocks <= 1024"
            + (
                "; msa_select_plan_wide: topk <= 64, B*H*n_blocks <= 4096"
                if wide
                else "; msa_select_plan_wide is off"
            )
            + "); using the unfused selection."
        )
    return None


@functools.lru_cache(maxsize=1)
def _on_gfx1250() -> bool:
    return "gfx1250" in get_hip_arch()


def _forced_of(indexer, S, NB, device):
    from primus.backends.megatron.core.transformer.minimax_m3.indexer import (
        _forced_blocks,
    )

    return _forced_blocks(S, NB, indexer.block_size, indexer.local_blocks, indexer.init_blocks, device)


def _fused_selection(
    raw, argmax, indexer, block_size, want_plan, bounded_chunks=False, wide=False, causal=False
):
    """:func:`select_plan`'s form of ``select_from_block_scores`` plus the ``_FusedSelect``
    bookkeeping and the :class:`BlockPlan` (``bounded_chunks``: the ``msa_bounded_plan_chunks``
    chunk size, as in ``build_block_plan``; ``wide``: the ``msa_select_plan_wide`` kernels).
    ``causal`` (``msa_select_skip_fill``): ``raw`` / ``argmax`` are unspecified on the blocks after
    each query's own; needs forced blocks, whose boost kernel then writes the -inf there."""
    B, H, S, NB = raw.shape
    topk = min(indexer.topk_blocks, NB)
    forced = _forced_of(indexer, S, NB, raw.device)
    assert forced is not None or not causal
    if forced is None:
        block_scores = raw
    else:
        block_scores = select_plan.boost_forced(raw, forced, causal_block=indexer.block_size if causal else 0)
    topk_scores, topk_idx = block_scores.topk(topk, dim=-1)
    bi, keys, keys_c, dead, table, plan = select_plan.select_post_and_plan(
        topk_scores, topk_idx, argmax, block_scores, block_size, want_plan, bounded_chunks, wide=wide
    )
    return bi, block_scores, keys, keys_c, dead, table, plan


def _unfused_selection(raw, argmax, select_from_block_scores, S):
    """The original selection and ``_FusedSelect`` bookkeeping: ``(block_indices, block_scores,
    keys, dead)``."""
    block_indices, block_scores = select_from_block_scores(raw, S)
    slots = block_indices.clamp_min(0)
    keys = argmax.gather(-1, slots)
    forced = torch.isposinf(block_scores.gather(-1, slots))
    dead = (block_indices < 0) | forced | (keys < 0)
    return block_indices, block_scores, keys, dead


def _live(values, n_live):
    """``values`` with its rows from ``n_live`` (a 0-dim tensor) on zeroed: the plan's spare
    entries / chunk rows are scratch."""
    at = torch.arange(values.shape[0], device=values.device)
    keep = at < n_live
    return torch.where(keep.view(-1, *([1] * (values.dim() - 1))), values, torch.zeros_like(values))


def _comparable(bi, block_scores, keys, keys_c, dead, table, plan):
    """What a selection hands on, as a list ``bitwise_check`` can compare."""
    return [
        bi,
        block_scores,
        keys,
        keys_c,
        dead,
        table,
        plan.ptr,
        _live(plan.entries, plan.ptr[-1]),
        plan.chunk_ptr,
        _live(plan.chunks, plan.chunk_ptr[-1]),
    ]


def _select_checked(name, shape_key, run_new, run_old) -> bool:
    """First-call self-check of a selection variant for a shape, on that call's own inputs
    (``bitwise_check.passed``; ``run_new`` / ``run_old`` return :func:`_comparable` lists)."""
    from primus.backends.megatron.core.fusions import bitwise_check

    return bitwise_check.passed(name, run_new, run_old, key=shape_key)


class _FusedSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, select_from_block_scores, block_size, plan_slot):
        indexer = getattr(select_from_block_scores, "__self__", None)
        config = getattr(indexer, "config", None)
        # msa_lean_masks: bit-identical scoring and dK kernels, each checked on its first call
        lean = bool(getattr(config, "msa_lean_masks", False))
        ctx.lean = lean
        ctx.dk_scatter = index_dk_scatter_enabled(config)
        ctx.bounded_chunks = bounded = bounded_plan_chunks_enabled(config)
        ctx.set_materialize_grads(False)
        ctx.block_size = block_size
        want_plan = bool(plan_slot) and plan_slot[0] is True  # fused_select_blocks: grad was on
        del plan_slot[:]

        # ---- eligibility, before any work ----
        S, B, H = q.shape[0], q.shape[1], q.shape[2]
        NB = -(-S // block_size)
        mode = _fused_mode(select_from_block_scores, B, H, S, NB)
        # the plan's chunk rule is part of what the self-checks compare, so it is part of their key
        shape_key = (S, B, H, NB, block_size, min(indexer.topk_blocks, NB), bounded) if mode else None
        # msa_select_skip_fill: with forced blocks the boost kernel rewrites the whole score table
        # and nothing else reads the blocks after a query's own, so the gfx1250 scoring kernel's
        # outputs need no -inf / -1 fill. Exact class: stays on under msa_exact_numerics.
        skip_fill = (
            mode is not None
            and bool(getattr(config, "msa_select_skip_fill", False))
            and _on_gfx1250()
            and _forced_of(indexer, S, NB, q.device) is not None
        )

        def scored(fill=True):
            return index_block_max(q, k, block_size, mask_skip=lean, fill=fill)

        def fused(fill, plan=want_plan, wide=None):
            raw, argmax = scored(fill)
            wide = mode == "wide" if wide is None else wide
            return _fused_selection(
                raw, argmax, indexer, block_size, plan, bounded, wide, causal=fill is not True
            )

        if mode == "wide":
            # the wide kernels against the unfused selection and build_block_plan (same chunk rule)

            def unfused_reference():
                raw, argmax = scored()
                bi, block_scores, keys, dead = _unfused_selection(raw, argmax, select_from_block_scores, S)
                plan = build_block_plan(bi, block_size, bounded)
                return _comparable(bi, block_scores, keys, keys.clamp_min(0), dead, bi.to(torch.int32), plan)

            if not _select_checked(
                "msa_select_plan_wide", shape_key, lambda: _comparable(*fused(True, True)), unfused_reference
            ):
                mode, skip_fill = None, False
        if skip_fill:
            # unwritten outputs poisoned, against the same selection on filled outputs
            skip_fill = _select_checked(
                "msa_select_skip_fill",
                shape_key,
                lambda: _comparable(*fused("poison", True)),
                lambda: _comparable(*fused(True, True)),
            )

        if mode is not None:
            block_indices, block_scores, keys, keys_c, dead, table, plan = fused(not skip_fill)
            ctx.save_for_backward(q, k, block_indices, keys, keys_c, dead)
            ctx.fused = True
            ctx.plan_slot = [plan] if plan is not None else []
            plan_slot.append((plan, table))
            ctx.mark_non_differentiable(block_indices)
            return block_indices, block_scores
        raw, argmax = scored()
        block_indices, block_scores, keys, dead = _unfused_selection(
            raw, argmax, select_from_block_scores, q.shape[0]
        )
        ctx.save_for_backward(q, k, block_indices, keys, dead)
        ctx.fused = False
        # filled by fused_select_blocks once the selection exists
        ctx.plan_slot = plan_slot
        ctx.mark_non_differentiable(block_indices)
        return block_indices, block_scores

    @staticmethod
    def backward(ctx, _grad_indices, grad_scores):
        if ctx.fused:
            q, k, block_indices, keys, keys_c, dead = ctx.saved_tensors
        else:
            q, k, block_indices, keys, dead = ctx.saved_tensors
        S, B, H, D = q.shape
        if grad_scores is None:
            return torch.zeros_like(q), torch.zeros_like(k), None, None, None

        if ctx.fused and grad_scores.dtype == torch.float32:
            g = select_plan.backward_g(grad_scores, block_indices, dead)
        else:
            g = (
                grad_scores.gather(-1, block_indices.clamp_min(0)).masked_fill_(dead, 0.0).float()
            )  # [B, H, S, topk]
            keys_c = keys.clamp_min(0)
        dq = index_dq(g, keys_c, k.reshape(S, B, D), H)
        plan = (
            ctx.plan_slot[0]
            if ctx.plan_slot
            else build_block_plan(block_indices, ctx.block_size, ctx.bounded_chunks)
        )
        dk = index_dk(plan, g, keys, q, ctx.block_size, lean=ctx.lean, scatter=ctx.dk_scatter)
        return dq, dk.reshape(k.shape), None, None, None


def fused_select_blocks(q, k, select_from_block_scores, block_size):
    """``(block_indices, block_scores, plan)``: the first two exactly as
    ``MinimaxM3Indexer.select_blocks``; ``plan`` is the selection's
    :class:`BlockPlan` when grad is enabled (a backward will need it), else None.

    Args:
        q: ``[S, B, n_index, D]`` bf16 index queries (normed, roped).
        k: ``[S, B, 1, D]`` bf16 index keys.
        select_from_block_scores: the indexer's boost + top-k.
    """
    plan_slot = [torch.is_grad_enabled()]
    block_indices, block_scores = _FusedSelect.apply(q, k, select_from_block_scores, block_size, plan_slot)
    if plan_slot:
        # fused selection: the plan (when grad is on) and the int32 table came out of the forward
        plan, table = plan_slot.pop()
        if plan is not None:
            plan = plan._replace(table=table)
        return block_indices, block_scores, plan
    plan = None
    if torch.is_grad_enabled():
        indexer = getattr(select_from_block_scores, "__self__", None)
        bounded = bounded_plan_chunks_enabled(getattr(indexer, "config", None))
        plan = build_block_plan(block_indices, block_size, bounded)
        plan_slot.append(plan)
    return block_indices, block_scores, plan
