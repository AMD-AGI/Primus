###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Exact-shape TP1 deferred CE, preserving the pinned TE BF16 rounding.

Forward retains exclusively owned logits and softmax statistics; backward
reuses logits storage. One backward per forward; no higher-order gradients.
Unsupported shapes/options retain TE. Set PRIMUS_LLAMA8B_DEFERRED_CE=0 to disable.
"""

import hashlib
import importlib
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from primus.core.patches import get_args, register_patch

SHAPE = (8192, 4, 128256)
BLOCK = 32768
WARPS = 16

# Derived from the pinned Transformer Engine Triton cross-entropy implementation.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


@triton.jit
def loss_stats_kernel(X, Y, STATS, LOSS, n_cols, IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    X += row * n_cols
    y = tl.load(Y + row)
    # Same FP32 online recurrence, exp and reduction geometry as pinned TE.
    m = float("-inf")
    d = 0.0
    for i in range(0, n_cols, BLOCK):
        offsets = i + tl.arange(0, BLOCK)
        x = tl.load(X + offsets, offsets < n_cols, other=float("-inf")).to(tl.float32)
        block_max = tl.max(x)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new
    tl.store(STATS + row * 2, m)
    tl.store(STATS + row * 2 + 1, d)
    xy = tl.load(X + y, (y >= 0) & (y < n_cols), other=float("-inf")).to(tl.float32)
    loss = -(xy - m - tl.log(d))
    tl.store(LOSS + row, tl.where(y == IGNORE, 0.0, loss))


@triton.jit
def deferred_grad_kernel(
    X, Y, STATS, UPSTREAM, n_cols, UPSTREAM_STRIDE: tl.constexpr, IGNORE: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    X += row * n_cols
    y = tl.load(Y + row)
    m = tl.load(STATS + row * 2)
    d = tl.load(STATS + row * 2 + 1)
    upstream = tl.load(UPSTREAM + row * UPSTREAM_STRIDE).to(tl.float32)
    for i in range(0, n_cols, BLOCK):
        offsets = i + tl.arange(0, BLOCK)
        x = tl.load(X + offsets, offsets < n_cols, other=float("-inf")).to(tl.float32)
        p = tl.exp(x - m) / d
        # Preserve TE's actual stored BF16 intermediate, including the second
        # rounding after the target subtraction. Do not algebraically combine.
        rounded_p = p.to(tl.bfloat16).to(tl.float32)
        target_grad = (rounded_p - 1.0).to(tl.bfloat16).to(tl.float32)
        unscaled = tl.where(offsets == y, target_grad, rounded_p)
        unscaled = tl.where(y == IGNORE, 0.0, unscaled)
        grad = (unscaled * upstream).to(tl.bfloat16)
        tl.store(X + offsets, grad, offsets < n_cols)


def eligible(
    inp,
    target,
    label_smoothing=0.0,
    reduce_loss=False,
    dist_process_group=None,
    ignore_idx=-100,
    is_cg_capturable=False,
):
    if not isinstance(inp, torch.Tensor) or not isinstance(target, torch.Tensor):
        return False
    if (
        not inp.is_cuda
        or not target.is_cuda
        or inp.device != target.device
        or inp.dtype != torch.bfloat16
        or target.dtype != torch.int64
        or tuple(inp.shape) != SHAPE
        or tuple(inp.stride()) != (4 * 128256, 128256, 1)
        or tuple(target.shape) != SHAPE[:2]
        or not target.is_contiguous()
        or label_smoothing != 0.0
        or reduce_loss
        or ignore_idx != -100
        or is_cg_capturable
    ):
        return False
    # Only the non-captured execution path is supported.
    if torch.cuda.is_current_stream_capturing():
        return False
    if dist_process_group is not None:
        if not torch.distributed.is_initialized():
            return False
        if torch.distributed.get_world_size(dist_process_group) != 1:
            return False
    return True


class _DeferredCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target):
        rows = SHAPE[0] * SHAPE[1]
        with torch.cuda.device(inp.device):
            stats = torch.empty((rows, 2), dtype=torch.float32, device=inp.device)
            loss = torch.empty(SHAPE[:2], dtype=torch.float32, device=inp.device)
            loss_stats_kernel[(rows,)](inp, target, stats, loss, SHAPE[2], -100, BLOCK, num_warps=WARPS)
        # Retaining the original BF16 logits costs the same buffer baseline TE
        # retains as its unscaled BF16 gradient. No full-size FP32 allocation.
        ctx.save_for_backward(inp.detach(), target, stats)
        ctx.consumed = False
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if ctx.consumed:
            raise RuntimeError("Deferred CE supports one backward per forward")
        inp, target, stats = ctx.saved_tensors
        if grad_output.dtype != torch.float32 or grad_output.device != inp.device:
            raise RuntimeError("Expected FP32 upstream gradient on the logits device")
        if tuple(grad_output.shape) != SHAPE[:2]:
            raise RuntimeError("Unexpected upstream gradient shape")
        # Expanded scalar gradients and transposed gradients must be materialized
        # in the same token order as TE, rather than interpreted as stride 1.
        grad_output = grad_output.contiguous()
        with torch.cuda.device(inp.device):
            deferred_grad_kernel[(SHAPE[0] * SHAPE[1],)](
                inp, target, stats, grad_output, SHAPE[2], 1, -100, BLOCK, num_warps=WARPS
            )
        ctx.consumed = True
        return inp, None


SOURCE_HASHES = {
    "transformer_engine/pytorch/cross_entropy.py": "6e58d905ff32967876ae7a30b29a343325bec498027f1b4895d2bff1e2c582ba",
    "transformer_engine/pytorch/triton/cross_entropy.py": "682082e251b1b1d335e28929390dfa32b99cd869c7377450535dd4d2112b5b5c",
    "transformer_engine/common/triton/cross_entropy.py": "7c57cb8f31788f8d1dacc2bc2a3a190b7d0bb024bf4332fd0aa51cd5691d6a62",
    "megatron/core/models/common/language_module/language_module.py": "7b09da6aad107851e47ffeecb87d18911b1a765410d7d988132ec39530f873eb",
    "megatron/core/extensions/transformer_engine.py": "26c8001d7454dc6c5b0715dd10436b9ec9966cc69c576e3a1a24fe7b2b158843",
}


def enabled(ctx, option, env_name):
    """Limit model defaults to the qualified MLPerf Llama recipe/runtime."""
    args = get_args(ctx)
    requested = os.getenv(env_name)
    if requested is None:
        requested = bool(getattr(args, option, False))
    else:
        requested = requested.lower() in ("1", "true")
    if not requested:
        return False
    expected = {
        "num_layers": 32,
        "hidden_size": 4096,
        "seq_length": 8192,
        "micro_batch_size": 4,
        "tensor_model_parallel_size": 1,
    }
    if any(getattr(args, key, None) != value for key, value in expected.items()):
        return False
    if not torch.cuda.is_available():
        return False
    device = torch.cuda.current_device()
    return (
        str(torch.__version__) == "2.12.0+rocm7.15.0a20260727"
        and torch.version.hip == "7.15.0"
        and torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] == "gfx950"
        and torch.cuda.get_device_name(device) == "AMD Instinct MI355X"
    )


@register_patch(
    "megatron.llama8b.deferred_ce",
    backend="megatron",
    phase="before_train",
    condition=lambda ctx: enabled(ctx, "llama8b_deferred_ce", "PRIMUS_LLAMA8B_DEFERRED_CE"),
    description="Defer exact-shape Llama 8B CE gradient construction into backward",
)
def install_deferred_ce(ctx):
    if (
        str(torch.__version__) != "2.12.0+rocm7.15.0a20260727"
        or torch.version.hip != "7.15.0"
        or triton.__version__ != "3.8.0"
        or torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] != "gfx950"
    ):
        raise RuntimeError("Llama 8B deferred CE requires the qualified MI355X runtime")
    for relative, expected in SOURCE_HASHES.items():
        module = importlib.import_module(relative[:-3].replace("/", "."))
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Unqualified CE source: " + relative)
    module = importlib.import_module("megatron.core.extensions.transformer_engine")
    original = module.parallel_cross_entropy
    if getattr(original, "_llama8b_deferred_ce", False):
        return

    def wrapped(
        inp,
        target,
        label_smoothing=0.0,
        reduce_loss=False,
        dist_process_group=None,
        ignore_idx=-100,
        is_cg_capturable=False,
        *,
        _input=None,
    ):
        if _input is None and eligible(
            inp, target, label_smoothing, reduce_loss, dist_process_group, ignore_idx, is_cg_capturable
        ):
            return _DeferredCE.apply(inp, target)
        return original(
            inp,
            target,
            label_smoothing,
            reduce_loss,
            dist_process_group,
            ignore_idx,
            is_cg_capturable,
            _input=_input,
        )

    wrapped._llama8b_deferred_ce = True
    module.parallel_cross_entropy = wrapped
