###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Defer TE fused cross entropy's gradient construction to the backward pass.

TE without ``overwrite_input`` builds the full logits gradient in the forward
pass. With ``deferred_cross_entropy`` the forward only computes the loss and two
FP32 statistics per token (running max and softmax denominator); the backward
rebuilds the softmax from the saved logits and writes the gradient over them
with the installed TE's rounding, so loss and gradient match it bit for bit:

* TE <= 2.17 (e.g. rocm/primus:v26.6) writes the gradient into the BF16 logits:
  the probability is rounded to BF16 and the target column rounded again after
  subtracting 1.
* TE with a separate FP32 gradient buffer (``cross_entropy_kernel`` takes a
  ``grad_input_ptr``) rounds once, after the upstream multiply in backward.

The first eligible call runs this path and the TE path it replaces (TE's
``overwrite_input`` one under ``te_ce_overwrite_input``, where the installed TE has it)
on a few positions of the real logits and as many wide random ones, and compares loss
and gradient bit for bit (``bitwise_check``); on a mismatch the run keeps TE. Like ``te_ce_overwrite_input``,
the logits hold their gradient after the loss backward.

Only TP=1, contiguous BF16 logits outside CUDA-graph capture take this path;
everything else calls the configured TE cross entropy.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from primus.backends.megatron.core.fusions import bitwise_check
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

IGNORE_INDEX = -100
# leading sequence positions the first call compares with TE
CHECK_ROWS = 16
MAX_BLOCK = 32768
NUM_WARPS = 16
# The gradient is elementwise, so its tile does not change any value; this one
# streams faster than TE's.
GRAD_BLOCK = 8192
GRAD_WARPS = 4


# Derived from the Transformer Engine Triton cross-entropy implementation.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
@triton.jit
def _loss_stats_kernel(X, Y, STATS, LOSS, n_cols, IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    X += row * n_cols
    y = tl.load(Y + row)
    # Same FP32 online recurrence, exp and reduction geometry as TE; the next
    # chunk's load is issued before the current chunk's reductions.
    m = float("-inf")
    d = 0.0
    offsets = tl.arange(0, BLOCK)
    x_next = tl.load(X + offsets, offsets < n_cols, other=float("-inf"))
    for i in range(0, n_cols, BLOCK):
        x = x_next.to(tl.float32)
        next_offsets = i + BLOCK + offsets
        x_next = tl.load(X + next_offsets, next_offsets < n_cols, other=float("-inf"))
        m_new = tl.maximum(m, tl.max(x))
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new
    tl.store(STATS + row * 2, m)
    tl.store(STATS + row * 2 + 1, d)
    xy = tl.load(X + y, (y >= 0) & (y < n_cols), other=float("-inf")).to(tl.float32)
    loss = -(xy - m - tl.log(d))
    tl.store(LOSS + row, tl.where(y == IGNORE, 0.0, loss))


@triton.jit
def _deferred_grad_kernel(
    X,
    Y,
    STATS,
    UPSTREAM,
    n_cols,
    IGNORE: tl.constexpr,
    ROUND_ONCE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    X += row * n_cols
    y = tl.load(Y + row)
    m = tl.load(STATS + row * 2)
    d = tl.load(STATS + row * 2 + 1)
    upstream = tl.load(UPSTREAM + row).to(tl.float32)
    for i in range(0, n_cols, BLOCK):
        offsets = i + tl.arange(0, BLOCK)
        x = tl.load(X + offsets, offsets < n_cols, other=float("-inf")).to(tl.float32)
        p = tl.exp(x - m) / d
        if ROUND_ONCE:
            target_grad = p - 1.0
        else:
            # Do not combine the two roundings algebraically.
            p = p.to(tl.bfloat16).to(tl.float32)
            target_grad = (p - 1.0).to(tl.bfloat16).to(tl.float32)
        unscaled = tl.where(offsets == y, target_grad, p)
        unscaled = tl.where(y == IGNORE, 0.0, unscaled)
        tl.store(X + offsets, (unscaled * upstream).to(tl.bfloat16), offsets < n_cols)


def te_rounds_once() -> Optional[bool]:
    """Whether the installed TE keeps the forward gradient in a separate FP32 buffer; None if
    its Triton kernel cannot be inspected."""
    try:
        from transformer_engine.common.triton.cross_entropy import cross_entropy_kernel
    except ImportError:
        return None
    arg_names = getattr(cross_entropy_kernel, "arg_names", None)
    if not arg_names:
        return None
    return "grad_input_ptr" in arg_names


class DeferredCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, round_once):
        vocab = logits.shape[-1]
        rows = target.numel()
        block = min(MAX_BLOCK, triton.next_power_of_2(vocab))
        stats = torch.empty((rows, 2), dtype=torch.float32, device=logits.device)
        loss = torch.empty(target.shape, dtype=torch.float32, device=logits.device)
        with torch.cuda.device(logits.device):
            _loss_stats_kernel[(rows,)](
                logits, target, stats, loss, vocab, IGNORE_INDEX, block, num_warps=NUM_WARPS
            )
        ctx.save_for_backward(logits.detach(), target, stats)
        ctx.round_once = bool(round_once)
        ctx.consumed = False
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if ctx.consumed:
            raise RuntimeError("deferred cross entropy supports one backward per forward")
        ctx.consumed = True
        logits, target, stats = ctx.saved_tensors
        grad_output = grad_output.float().contiguous()
        with torch.cuda.device(logits.device):
            _deferred_grad_kernel[(target.numel(),)](
                logits,
                target,
                stats,
                grad_output,
                logits.shape[-1],
                IGNORE_INDEX,
                ctx.round_once,
                min(GRAD_BLOCK, triton.next_power_of_2(logits.shape[-1])),
                num_warps=GRAD_WARPS,
            )
        torch.autograd.graph.increment_version(logits)
        return logits, None, None


def eligible(logits, labels, tp_group, is_cg_capturable) -> bool:
    return (
        not is_cg_capturable
        and logits.is_cuda
        and logits.dtype == torch.bfloat16
        and logits.is_contiguous()
        and labels.dtype == torch.int64
        and labels.is_contiguous()
        and tuple(labels.shape) == tuple(logits.shape[:-1])
        and (tp_group is None or torch.distributed.get_world_size(tp_group) == 1)
        and not torch.cuda.is_current_stream_capturing()
    )


def _matches_original(original, logits, labels, tp_group, round_once) -> bool:
    def inputs():
        """The leading positions of the real logits (near uniform early in training), then as
        many of wide random logits with one ignored label."""
        x, y = logits[:CHECK_ROWS], labels[:CHECK_ROWS]
        wide = (4 * bitwise_check.randn(x.shape, x.device)).to(x.dtype)
        y_ignored = y.clone()
        y_ignored.view(-1)[0] = IGNORE_INDEX
        return torch.cat([x, wide]), torch.cat([y, y_ignored])

    return bitwise_check.passed(
        "deferred_cross_entropy",
        lambda: bitwise_check.outputs_and_grads(
            lambda x, y: DeferredCrossEntropy.apply(x, y, round_once), *inputs()
        ),
        lambda: bitwise_check.outputs_and_grads(lambda x, y: original(x, y, tp_group, False), *inputs()),
        key=logits.shape[-1],
    )


def make_deferred_cross_entropy(original, round_once=None):
    """Megatron's ``te_parallel_cross_entropy`` with the gradient deferred to backward."""
    if round_once is None:
        round_once = te_rounds_once()
    if round_once is None:
        raise RuntimeError("cannot tell how the installed TE cross entropy rounds its gradient")

    def te_parallel_cross_entropy(logits, labels, tp_group, is_cg_capturable=False):
        if not (
            eligible(logits, labels, tp_group, is_cg_capturable)
            and _matches_original(original, logits, labels, tp_group, round_once)
        ):
            return original(logits, labels, tp_group, is_cg_capturable)
        return DeferredCrossEntropy.apply(logits, labels, round_once)

    te_parallel_cross_entropy._primus_deferred_ce = True
    return te_parallel_cross_entropy


def _enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return (
        bool(getattr(args, "deferred_cross_entropy", False))
        and bool(getattr(args, "cross_entropy_loss_fusion", False))
        and getattr(args, "cross_entropy_fusion_impl", None) == "te"
    )


@register_patch(
    "megatron.te.cross_entropy_deferred",
    backend="megatron",
    phase="before_train",
    description="Defer TE fused cross entropy's gradient to backward (TP=1 BF16)",
    condition=_enabled,
    # After megatron.te.cross_entropy_overwrite_input, so ineligible calls reach it.
    priority=60,
)
def patch_te_cross_entropy_deferred(ctx: PatchContext) -> None:
    # validate_args_on_rocm rejects use_turbo_cross_entropy with this patch.
    import megatron.core.extensions.transformer_engine as te_ext
    from megatron.core.models.common.language_module import language_module

    original = te_ext.te_parallel_cross_entropy
    if original is None or getattr(original, "_primus_deferred_ce", False):
        return
    round_once = te_rounds_once()
    if round_once is None:
        warning_rank_0(
            "[Patch:megatron.te.cross_entropy_deferred] skipped: cannot inspect TE's Triton "
            "cross_entropy_kernel to match its gradient rounding; TE cross entropy unchanged"
        )
        return
    wrapped = make_deferred_cross_entropy(original, round_once)
    te_ext.te_parallel_cross_entropy = wrapped
    language_module.te_parallel_cross_entropy = wrapped
    log_rank_0(
        "[Patch:megatron.te.cross_entropy_deferred] TE cross entropy builds its gradient "
        f"in backward over the logits (TP=1 BF16, rounded {'once' if round_once else 'twice'} "
        "like the installed TE and checked against it on the first call; other cases keep TE)"
    )
