# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Compile-friendly MXFP6 (E2M3) linear layers for Megatron local spec.

The MXFP4 sibling of this module (``primus_turbo_mxfp4_local``) is the template, and
this one is deliberately smaller, because MXFP6 removes most of MXFP4's configuration
surface rather than because anything is left unfinished:

- **No preshuffle contract.** The A6W6 kernels read AITER's packed C0/C1 tile blob
  directly, so there is no unshuffled layout and no fast path to opt into. MXFP4's
  ``_enable_preshuffle`` / ``_assert_preshuffle_contract`` dance, and the whole class of
  misconfiguration it guards against, simply does not exist here.
- **No ScalingRecipe flags.** The 32-point Hadamard rotation is mandatory and fused into
  the packer (the GEMM depends on it cancelling between the two operands), scaling is
  strictly per-1x32 along the contraction axis so ``use_2d_block`` is meaningless, and
  stochastic rounding is not implemented. MXFP4 threads twelve booleans through its
  quantize op; MXFP6 has none to thread.
- **No local custom-op registration.** Primus-Turbo already exposes
  ``primus_turbo::quantize_mxfp6_dual_impl`` as a ``torch.library.custom_op`` with a
  correct fake, so unlike MXFP4 there is nothing to re-wrap in order to bypass
  recipe construction.

Retained from the MXFP4 design: the ``setup_context`` pattern with primitive-only
arguments so ``torch.compile`` traces without graph breaks, the two backward modes
(pure MXFP6, or hybrid MXFP6-forward / FP8-backward), and zero TransformerEngine
dependencies.

Shape constraint worth knowing: MXFP6 needs the linear's M, N **and** K to be multiples
of 256. K is included because the backward GEMMs use it as an output dimension. This is
enforced inside Primus-Turbo's ``gemm_fp6``; here it means a hidden size or sequence
length that is only 128-aligned will be rejected at the first forward.
"""

import functools
import os
import warnings

import torch
import torch.nn.functional as F
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLP
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    MXFP6_PROLOGUE_BIAS_GELU,
    MXFP6_PROLOGUE_BIAS_GELU_BACKWARD,
    ScalingGranularity,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import gemm_fp6_impl, gemm_fp6_out_impl
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import check_mxfp6_support

from .primus_turbo_float8_local import _quantize_fp8_tw

_GRAN_VALUE = ScalingGranularity.MX_BLOCKWISE.value

# Registered by Primus-Turbo with a pure-arithmetic fake, so it is safe to trace.
_quantize_mxfp6_dual = torch.ops.primus_turbo.quantize_mxfp6_dual_impl
_quantize_mxfp6_fused_dual = torch.ops.primus_turbo.quantize_mxfp6_fused_dual_impl


def _wgrad_into_main_grad(weight, g_col, g_col_scale, a_col, a_col_scale, n, k, m):
    """Store the weight gradient directly into ``weight.main_grad``.

    Saves the round trip the unfused path forces: a freshly allocated wgrad, handed to
    autograd as ``weight.grad``, which Megatron's DDP backward hook then adds into
    ``main_grad`` and frees. Here the A6W6 asm writes ``main_grad`` itself and the hook
    has nothing left to do.

    Correct only because the store has beta=0 and there is exactly one microbatch per
    optimizer step; ``_init_mxfp6_linear`` enforces that.

    ``grad_added_to_main_grad`` is *not* set here -- see ``_claim_main_grad``, which has to
    do it from the forward instead.

    Returns the placeholder that has to go back to autograd as the weight gradient. It is
    never read: with ``grad_added_to_main_grad`` set, the hook skips its ``add_`` and
    immediately drops ``param.grad``. It exists only because the hook asserts
    ``param.grad is not None`` whenever ``overlap_grad_reduce`` is on. This mirrors what
    Megatron's own ``gradient_accumulation_fusion`` path returns. Note for plan item 0.2b:
    this allocation sits inside what a full-iteration CUDA graph would capture.
    """
    main_grad = getattr(weight, "main_grad", None)
    if main_grad is None:
        raise RuntimeError(
            "MXFP6 wgrad fusion needs weight.main_grad, which Megatron DDP allocates. "
            "Either wrap the model in DistributedDataParallel or set "
            "mxfp6_fused_wgrad_accum=False."
        )
    if main_grad.dtype != torch.bfloat16:
        raise TypeError(
            f"MXFP6 wgrad fusion writes bf16 only, but main_grad is {main_grad.dtype}. "
            "Set main_grads_dtype=bf16 or mxfp6_fused_wgrad_accum=False."
        )

    gemm_fp6_out_impl(g_col, g_col_scale, a_col, a_col_scale, main_grad, n, k, m, _GRAN_VALUE)
    return torch.empty_like(weight)


def _claim_main_grad(*weights) -> None:
    """Tell Megatron's DDP hook that these weights' gradients are already in main_grad.

    Called from the forward, which reads oddly, because the natural place -- right after
    the backward's store -- is not available: dynamo refuses to trace a mutation of state
    owned outside an autograd.Function ("HOP: Unsafe side effect"), and rather than fail it
    breaks the graph around every MXFP6 linear. On the MBS=32 Flux 12B arm that fragmented
    one compiled block into hundreds and cost 42.7 ms of eager elementwise work per 512
    images, against the ~16 ms of ``add_`` the fusion removes -- a net regression. From the
    forward the same assignment is ordinary traced code, which dynamo records as a side
    effect and replays.

    Setting it before the store rather than after is safe in one direction only, and this
    is that direction: the flag is read by the DDP backward hook, which cannot run until
    the backward has stored, and ``zero_grad_buffer`` clears it at the top of every step.
    A forward with no backward (eval, ``no_grad``) leaves it set with no hook to read it.

    Megatron does the same thing for its own reason -- ``zero_grad_buffer`` skips the reset
    under TE CUDA graphs precisely because the capture "no longer has the opportunity to
    set it back to True".
    """
    for weight in weights:
        weight.grad_added_to_main_grad = True


class MXFP6LinearFunction(torch.autograd.Function):
    """MXFP6 linear (Y = X @ W^T) with MX block-of-32 scaling along the contraction axis.

    Two modes via the ``backward_is_fp8`` bool primitive:

    - Pure MXFP6: forward and backward both quantize to MXFP6 and call gemm_fp6_impl.
    - Hybrid: forward is MXFP6, backward re-quantizes the saved BF16 to tensorwise FP8.

    In the pure path the forward returns the column-direction blobs as extra outputs so
    ``setup_context`` can save them; they are already uint8, so unlike MXFP4 there is no
    dtype-view juggling needed to keep the autograd engine from trying to allocate zero
    gradients in an unsupported dtype.
    """

    @staticmethod
    def forward(
        input,
        weight,
        backward_is_fp8,
        fp8_bwd_dtype,
        fp8_gran_value,
        fp8_backend_value,
        fuse_wgrad_accum,
    ):
        out_dtype = input.dtype
        orig_shape = input.shape
        input_2d = input.reshape(-1, input.shape[-1])

        m, k = input_2d.shape
        n = weight.shape[0]

        a_row, a_row_scale, a_col, a_col_scale = _quantize_mxfp6_dual(input_2d)
        b_row, b_row_scale, b_col, b_col_scale = _quantize_mxfp6_dual(weight)

        output = gemm_fp6_impl(
            a_row,
            a_row_scale,
            b_row,
            b_row_scale,
            m,
            n,
            k,
            out_dtype,
            _GRAN_VALUE,
        )
        output = output.reshape(*orig_shape[:-1], output.shape[-1])

        if backward_is_fp8:
            return output, input_2d.view_as(input_2d), weight.view_as(weight)
        return output, a_col, a_col_scale, b_col, b_col_scale

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            input,
            weight,
            backward_is_fp8,
            fp8_bwd_dtype,
            fp8_gran_value,
            fp8_backend_value,
            fuse_wgrad_accum,
        ) = inputs

        ctx.backward_is_fp8 = backward_is_fp8
        ctx.fuse_wgrad_accum = fuse_wgrad_accum
        ctx.out_dtype = input.dtype
        ctx.orig_shape = input.shape
        # The packed blobs carry no shape, so the logical dims have to be saved too.
        ctx.m = input.numel() // input.shape[-1]
        ctx.k = input.shape[-1]
        ctx.n = weight.shape[0]

        if backward_is_fp8:
            _, input_2d_saved, weight_saved = output
            ctx.save_for_backward(input_2d_saved, weight_saved)
            ctx.fp8_bwd_dtype = fp8_bwd_dtype
            ctx.fp8_gran_value = fp8_gran_value
            ctx.fp8_backend_value = fp8_backend_value
        else:
            _, a_col, a_col_scale, b_col, b_col_scale = output
            # The weight rides along only when the backward has to reach through it to
            # weight.main_grad; save_for_backward hands back the same Parameter object.
            if fuse_wgrad_accum:
                ctx.save_for_backward(a_col, a_col_scale, b_col, b_col_scale, weight)
            else:
                ctx.save_for_backward(a_col, a_col_scale, b_col, b_col_scale)
            ctx.mark_non_differentiable(a_col, a_col_scale, b_col, b_col_scale)

    @staticmethod
    def backward(ctx, grad_output, *_):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        m, n, k = ctx.m, ctx.n, ctx.k

        if ctx.backward_is_fp8:
            input_2d, weight = ctx.saved_tensors

            grad_fp8, grad_scale_inv = _quantize_fp8_tw(grad_2d, ctx.fp8_bwd_dtype)
            a_fp8, a_scale_inv = _quantize_fp8_tw(input_2d, ctx.fp8_bwd_dtype)
            b_fp8, b_scale_inv = _quantize_fp8_tw(weight, ctx.fp8_bwd_dtype)

            grad_input = gemm_fp8_impl(
                grad_fp8,
                grad_scale_inv,
                False,
                b_fp8,
                b_scale_inv,
                False,
                ctx.out_dtype,
                False,
                granularity=ctx.fp8_gran_value,
                default_backend=ctx.fp8_backend_value,
            )
            grad_input = grad_input.reshape(ctx.orig_shape)

            grad_weight = gemm_fp8_impl(
                a_fp8,
                a_scale_inv,
                True,
                grad_fp8,
                grad_scale_inv,
                False,
                ctx.out_dtype,
                True,
                granularity=ctx.fp8_gran_value,
                default_backend=ctx.fp8_backend_value,
            )
        else:
            if ctx.fuse_wgrad_accum:
                a_col, a_col_scale, b_col, b_col_scale, weight = ctx.saved_tensors
            else:
                a_col, a_col_scale, b_col, b_col_scale = ctx.saved_tensors
            g_row, g_row_scale, g_col, g_col_scale = _quantize_mxfp6_dual(grad_2d)

            # grad_input[M, K] = grad[M, N] @ weight[N, K], contracting N. b_col is the
            # weight packed along N, i.e. logically [K, N] contracting N.
            grad_input = gemm_fp6_impl(
                g_row,
                g_row_scale,
                b_col,
                b_col_scale,
                m,
                k,
                n,
                ctx.out_dtype,
                _GRAN_VALUE,
            )
            grad_input = grad_input.reshape(ctx.orig_shape)

            # grad_weight[N, K] = grad.T[N, M] @ input[M, K], contracting M.
            if ctx.fuse_wgrad_accum:
                grad_weight = _wgrad_into_main_grad(
                    weight, g_col, g_col_scale, a_col, a_col_scale, n, k, m
                )
            else:
                grad_weight = gemm_fp6_impl(
                    g_col,
                    g_col_scale,
                    a_col,
                    a_col_scale,
                    n,
                    k,
                    m,
                    ctx.out_dtype,
                    _GRAN_VALUE,
                )

        return grad_input, grad_weight, None, None, None, None, None


def _resolve_wgrad_fusion(module, name: str) -> bool:
    """Whether this linear may write its wgrad straight into ``main_grad``.

    Gated on the Primus-owned ``mxfp6_fused_wgrad_accum`` rather than Megatron's
    ``gradient_accumulation_fusion``, because the latter also moves every plain linear --
    see the field's own comment for what that costs Flux's AdaLN projections.

    Two hard requirements, both raised rather than silently downgraded so that a
    misconfiguration does not read as a performance result:

    - One microbatch per optimizer step. The A6W6 asm stores with beta=0, so a second
      microbatch would overwrite the first one's gradient instead of adding to it.
      Megatron's own fused path uses a beta=1 accumulate kernel and has no such limit.
    - The pure-MXFP6 backward. The FP8-backward mode forms its wgrad with
      ``gemm_fp8_impl``, which has no caller-provided-output variant.
    """
    if not getattr(module.config, "mxfp6_fused_wgrad_accum", False):
        return False

    if module._backward_is_fp8:
        raise ValueError(
            f"{name} cannot combine mxfp6_fused_wgrad_accum=True with "
            "mxfp6_backward_precision='fp8': the FP8 backward has no out-variant GEMM to "
            "write main_grad with. Use mxfp6_backward_precision='mxfp6' or turn the "
            "fusion off."
        )

    try:
        from megatron.core.num_microbatches_calculator import get_num_microbatches

        num_microbatches = get_num_microbatches()
    except (ImportError, AttributeError):
        # Calculator not up yet (unit tests build these modules standalone). The config
        # is the only authority available, and the store stays correct as long as the
        # caller honours the one-microbatch rule.
        num_microbatches = None

    if num_microbatches is not None and num_microbatches > 1:
        raise ValueError(
            f"{name} requires mxfp6_fused_wgrad_accum=False when there is more than one "
            f"microbatch per step (got {num_microbatches}). The A6W6 store has no beta=1 "
            "accumulate epilogue, so it would overwrite earlier microbatches."
        )

    return True


def _init_mxfp6_linear(module) -> None:
    """Shared __init__ tail for both MXFP6 parallel linears.

    MXFP4 duplicates this block between its column and row classes; there is no reason
    for the MXFP6 copy to inherit the duplication.
    """
    name = type(module).__name__

    if module.config.tensor_model_parallel_size != 1:
        raise ValueError(
            f"{name} requires tensor_model_parallel_size=1. "
            f"Got {module.config.tensor_model_parallel_size}."
        )
    if module.gradient_accumulation_fusion:
        # Megatron's fused path needs a beta=1 accumulate epilogue, which the A6W6 entry
        # point has not got. The MXFP6 equivalent is mxfp6_fused_wgrad_accum, which does
        # not disturb the plain linears.
        raise ValueError(
            f"{name} requires gradient_accumulation_fusion=False. To fuse the MXFP6 "
            "weight gradient into main_grad, set mxfp6_fused_wgrad_accum=True instead."
        )
    if module.sequence_parallel:
        raise ValueError(f"{name} requires sequence_parallel=False.")

    supported, reason = check_mxfp6_support()
    if not supported:
        raise RuntimeError(f"MXFP6 not supported on this device: {reason}")

    module._backward_is_fp8 = getattr(module.config, "mxfp6_backward_precision", "mxfp6") == "fp8"
    module._fuse_wgrad_accum = _resolve_wgrad_fusion(module, name)

    if module._backward_is_fp8:
        from primus_turbo.pytorch.core.low_precision import float8_e5m2

        module._fp8_bwd_dtype = float8_e5m2
        module._fp8_gran_value = ScalingGranularity.TENSORWISE.value
        module._fp8_backend_value = BackendType.HIPBLASLT.value
    else:
        module._fp8_bwd_dtype = None
        module._fp8_gran_value = 0
        module._fp8_backend_value = 0


def _mxfp6_forward_impl(module, input, weight, **kwargs):
    bias = kwargs.get("bias", None)

    if module._fuse_wgrad_accum:
        _claim_main_grad(weight)

    result = MXFP6LinearFunction.apply(
        input,
        weight,
        module._backward_is_fp8,
        module._fp8_bwd_dtype,
        module._fp8_gran_value,
        module._fp8_backend_value,
        module._fuse_wgrad_accum,
    )
    output = result[0]

    # Bias is added outside the GEMM: the A6W6 entry point has no bias epilogue.
    if bias is not None:
        output = output + bias
    return output


class MXFP6ColumnParallelLinear(ColumnParallelLinear):
    """ColumnParallelLinear with per-module MXFP6. torch.compile friendly.

    Requires: tensor_model_parallel_size=1, gradient_accumulation_fusion=False,
    sequence_parallel=False. ``mxfp6_fused_wgrad_accum=True`` is supported at one
    microbatch per step; see ``_resolve_wgrad_fusion``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _init_mxfp6_linear(self)

    def _forward_impl(self, input, weight, *args, **kwargs):
        return _mxfp6_forward_impl(self, input, weight, **kwargs)


class MXFP6RowParallelLinear(RowParallelLinear):
    """RowParallelLinear with per-module MXFP6. torch.compile friendly.

    Requires: tensor_model_parallel_size=1, gradient_accumulation_fusion=False,
    sequence_parallel=False. ``mxfp6_fused_wgrad_accum=True`` is supported at one
    microbatch per step; see ``_resolve_wgrad_fusion``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _init_mxfp6_linear(self)

    def _forward_impl(self, input, weight, *args, **kwargs):
        return _mxfp6_forward_impl(self, input, weight, **kwargs)


# ---------------------------------------------------------------------------
# Whole-MLP fusion.
#
# Splitting the MLP into two independent autograd Functions forces the activation to
# exist: fc2's forward has to receive a real tensor, and fc1's backward has to receive
# one. Owning fc1 -> epilogue -> fc2 in a single Function is what lets the packer take
# the epilogue as a prologue instead, in both directions -- the activation is packed
# straight out of LDS and the pre-activation gradient is never assembled at all.
#
# What that removes, per Flux 12B step at the profiled shapes: the bias-add + GELU kernel's
# read and write and the packer's read back of it in the forward, and the same round-trip
# plus the bias gradient's own reduction pass in the backward. Measured on 8x MI355X at
# micro_batch_size 64, against this same module with the fusion switched off: 75.3 ms/step of
# epilogue and reduction kernels go away, 35.5 ms/step of added prologue cost inside the
# packer replaces them, and the step's GPU busy time falls 856.5 -> 810.9 ms, 5.0% off wall
# clock (863.6 -> 820.3 ms/step). The step is GPU bound at 96.8% busy, so that lands as
# throughput: 74.1 -> 78.0 images/s/GPU. The pre-activation y1 is still saved, but it was
# already being saved for the activation's own backward, so peak allocated memory only grows
# by the column-sum buffer, 4 MB of 244 GB. Reserved memory grows more, 249.0 -> 250.2 GB,
# because the freed epilogue temporaries leave differently shaped holes in the caching
# allocator; that is the number the driver reports, so it is what a memory ceiling will see.
#
# The backward is where the win is, ~0.40 ms per call against ~0.11 for the forward, and the
# reason is worth knowing before trying to improve this. The packer is bandwidth bound at
# 3.8 TB/s without a prologue, so fusing work into it only pays while it stays that way. The
# forward prologue removes a 0.26 ms kernel and adds 0.14 ms of arithmetic to a 0.38 ms pack;
# the backward removes two kernels totalling 0.73 ms and its extra read of the incoming
# gradient is traffic it would have done anyway. An early version of the prologue used a libm
# tanh and a per-element bounds branch and cost 0.38 ms of arithmetic instead of 0.14, which
# made the forward a net regression and cost most of the win.
# ---------------------------------------------------------------------------


class MXFP6MLPFunction(torch.autograd.Function):
    """fc1 GEMM, bias+GELU, fc2 GEMM as one op, with the activation never in HBM.

    Follows ``MXFP6LinearFunction``'s conventions: the column-direction blobs leave as
    extra outputs so ``setup_context`` can save them and mark them non-differentiable, and
    all non-tensor state lands on ``ctx`` as primitives so ``torch.compile`` traces cleanly.

    Only the pure-MXFP6 backward is supported. The FP8-backward mode re-quantizes saved
    BF16 activations, which would put the activation back in HBM and defeat the point;
    ``_fused_mlp_unusable_reason`` rejects that configuration before we get here.
    """

    @staticmethod
    def forward(hidden_states, w1, b1, w2, fuse_wgrad_accum):
        out_dtype = hidden_states.dtype
        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, orig_shape[-1])

        m, k = x.shape
        f = w1.shape[0]
        h = w2.shape[0]

        x_row, x_row_s, x_col, x_col_s = _quantize_mxfp6_dual(x)
        w1_row, w1_row_s, w1_col, w1_col_s = _quantize_mxfp6_dual(w1)

        # Pre-activation. Saved for backward, where the epilogue is recomputed from it
        # rather than its output being stashed -- the same bytes are held either way.
        y1 = gemm_fp6_impl(x_row, x_row_s, w1_row, w1_row_s, m, f, k, out_dtype, _GRAN_VALUE)

        # gelu(y1 + b1), packed in both directions without ever being written out.
        a_row, a_row_s, a_col, a_col_s, _ = _quantize_mxfp6_fused_dual(
            y1, None, b1, MXFP6_PROLOGUE_BIAS_GELU, False
        )
        w2_row, w2_row_s, w2_col, w2_col_s = _quantize_mxfp6_dual(w2)

        output = gemm_fp6_impl(a_row, a_row_s, w2_row, w2_row_s, m, h, f, out_dtype, _GRAN_VALUE)
        output = output.reshape(*orig_shape[:-1], h)

        return output, y1, x_col, x_col_s, a_col, a_col_s, w1_col, w1_col_s, w2_col, w2_col_s

    @staticmethod
    def setup_context(ctx, inputs, output):
        hidden_states, w1, b1, w2, fuse_wgrad_accum = inputs

        ctx.fuse_wgrad_accum = fuse_wgrad_accum
        ctx.out_dtype = hidden_states.dtype
        ctx.orig_shape = hidden_states.shape
        # The packed blobs carry no shape, so the logical dims have to be saved too.
        ctx.m = hidden_states.numel() // hidden_states.shape[-1]
        ctx.k = hidden_states.shape[-1]
        ctx.f = w1.shape[0]
        ctx.h = w2.shape[0]

        _, y1, x_col, x_col_s, a_col, a_col_s, w1_col, w1_col_s, w2_col, w2_col_s = output
        blobs = (x_col, x_col_s, a_col, a_col_s, w1_col, w1_col_s, w2_col, w2_col_s)
        # b1 is a leaf parameter, so saving it costs nothing, and the backward needs it to
        # rebuild the pre-activation for the GELU derivative. w1 and w2 ride along only
        # when the backward has to reach through them to their main_grad buffers.
        extra = (w1, w2) if fuse_wgrad_accum else ()
        ctx.save_for_backward(y1, b1, *blobs, *extra)
        ctx.mark_non_differentiable(*blobs)

    @staticmethod
    def backward(ctx, grad_output, *_):
        (
            y1,
            b1,
            x_col,
            x_col_s,
            a_col,
            a_col_s,
            w1_col,
            w1_col_s,
            w2_col,
            w2_col_s,
            *fused_weights,
        ) = ctx.saved_tensors
        m, k, f, h = ctx.m, ctx.k, ctx.f, ctx.h
        out_dtype = ctx.out_dtype

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        g2 = grad_output.reshape(-1, h)

        g2_row, g2_row_s, g2_col, g2_col_s = _quantize_mxfp6_dual(g2)

        # fc2 dgrad: [m, f] = g2[m, h] @ w2[h, f], contracting h.
        grad_a = gemm_fp6_impl(g2_row, g2_row_s, w2_col, w2_col_s, m, f, h, out_dtype, _GRAN_VALUE)
        # fc2 wgrad: [h, f] = g2.T[h, m] @ a[m, f], contracting m.
        if ctx.fuse_wgrad_accum:
            grad_w2 = _wgrad_into_main_grad(
                fused_weights[1], g2_col, g2_col_s, a_col, a_col_s, h, f, m
            )
        else:
            grad_w2 = gemm_fp6_impl(g2_col, g2_col_s, a_col, a_col_s, h, f, m, out_dtype, _GRAN_VALUE)

        # The GELU derivative is applied while staging, so grad_y1 is never assembled. Its
        # column sums come back as a side output because the bias gradient is a reduction
        # over exactly the tensor that no longer exists.
        want_bias_grad = ctx.needs_input_grad[2]
        g1_row, g1_row_s, g1_col, g1_col_s, b1_partial = _quantize_mxfp6_fused_dual(
            y1, grad_a, b1, MXFP6_PROLOGUE_BIAS_GELU_BACKWARD, want_bias_grad
        )

        # fc1 dgrad: [m, k] = grad_y1[m, f] @ w1[f, k], contracting f.
        grad_x = gemm_fp6_impl(g1_row, g1_row_s, w1_col, w1_col_s, m, k, f, out_dtype, _GRAN_VALUE)
        grad_x = grad_x.reshape(ctx.orig_shape)
        # fc1 wgrad: [f, k] = grad_y1.T[f, m] @ x[m, k], contracting m.
        if ctx.fuse_wgrad_accum:
            grad_w1 = _wgrad_into_main_grad(
                fused_weights[0], g1_col, g1_col_s, x_col, x_col_s, f, k, m
            )
        else:
            grad_w1 = gemm_fp6_impl(g1_col, g1_col_s, x_col, x_col_s, f, k, m, out_dtype, _GRAN_VALUE)

        grad_b1 = b1_partial.sum(0).to(out_dtype) if want_bias_grad else None

        return grad_x, grad_w1, grad_b1, grad_w2, None


def _is_tanh_gelu(fn) -> bool:
    """Whether ``fn`` is exactly ``F.gelu(approximate="tanh")``.

    The packer's prologue implements that function and only that one, to within a rounding of
    the tanh. This check is not paranoia: ``FluxConfig``'s default activation is
    ``openai_gelu_no_jit``, the same mathematical function written as
    ``beta * x * (1 + kappa * x^2)`` rather than ``beta * (x + kappa * x^3)``. Those disagree
    well above the prologue's own rounding, and the YAML key that selects between them
    (``activation_func: openai_gelu`` maps to the fused ATen one)
    makes it easy to land on the other branch without noticing.
    """
    if isinstance(fn, functools.partial):
        return fn.func is F.gelu and fn.keywords.get("approximate") == "tanh"
    return False


def _fused_mlp_unusable_reason(mlp) -> str:
    """Why the fused MLP cannot be used for this module, or ``""`` if it can.

    Every branch corresponds to a path in ``MLP.forward`` that the fused Function does not
    reproduce. Returning a reason rather than silently deferring keeps a misconfiguration
    from looking like a performance result.
    """
    config = mlp.config

    if not hasattr(torch.ops.primus_turbo, "quantize_mxfp6_fused_dual_impl"):
        return "this Primus-Turbo build has no fused MXFP6 prologue packer"
    if config.gated_linear_unit:
        return "gated_linear_unit splits the fc1 output, which the prologue does not do"
    if getattr(config, "bias_activation_fusion", False):
        return "bias_activation_fusion routes the epilogue through Megatron's own fused kernel"
    if getattr(config, "use_te_activation_func", False):
        return "use_te_activation_func replaces the activation with a TE module"
    if not _is_tanh_gelu(mlp.activation_func):
        name = getattr(mlp.activation_func, "__name__", type(mlp.activation_func).__name__)
        return (
            "the fused prologue implements F.gelu(approximate='tanh') only, but the "
            f"activation is {name!r}"
        )

    for name in ("linear_fc1", "linear_fc2"):
        linear = getattr(mlp, name)
        if not isinstance(linear, (MXFP6ColumnParallelLinear, MXFP6RowParallelLinear)):
            return f"{name} is {type(linear).__name__}, not an MXFP6 linear"
        if getattr(linear, "_backward_is_fp8", False):
            return (
                "mxfp6_backward_precision='fp8' saves the activation for backward "
                "requantization, which the fusion removes"
            )
        if not linear.skip_bias_add:
            return f"{name} adds its own bias, so the epilogue is not the MLP's to fuse"

    return ""


def _fused_mlp_mode() -> str:
    mode = os.environ.get("PRIMUS_MXFP6_FUSED_MLP", "").strip().lower() or "auto"
    assert mode in (
        "auto",
        "on",
        "off",
    ), f"PRIMUS_MXFP6_FUSED_MLP must be auto, on or off, got {mode!r}"
    return mode


class MXFP6FusedMLP(MLP):
    """MLP whose bias-add + GELU is folded into the MXFP6 packer, in both directions.

    Drop-in for ``MLP``: same submodules, same ``(output, output_bias)`` return, same
    parameters and state dict. Only ``forward`` differs, and it defers to ``MLP.forward``
    for anything the fused path does not cover, so a configuration it cannot handle is
    slow rather than wrong.

    ``PRIMUS_MXFP6_FUSED_MLP=off`` forces the stock path for A/B comparison; ``on`` makes
    an unusable configuration an error instead of a silent fallback.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mode = _fused_mlp_mode()
        reason = "disabled by environment" if mode == "off" else _fused_mlp_unusable_reason(self)
        self._fused_epilogue = reason == ""

        if not self._fused_epilogue and mode == "on":
            raise RuntimeError(f"PRIMUS_MXFP6_FUSED_MLP=on but the fused MLP is unusable: {reason}")
        if not self._fused_epilogue and mode == "auto":
            warnings.warn(
                f"MXFP6 fused MLP epilogue disabled, falling back to the stock MLP: {reason}",
                stacklevel=2,
            )

    def forward(self, hidden_states, per_token_scale=None):
        # per_token_scale scales the activation after the epilogue, which is exactly the
        # tensor the fusion refuses to materialise. Only MoE experts pass it.
        if not self._fused_epilogue or per_token_scale is not None:
            return super().forward(hidden_states, per_token_scale=per_token_scale)

        # Both linears resolve the flag from the same config field, so fc1's answer covers
        # fc2's weight too.
        fuse_wgrad_accum = self.linear_fc1._fuse_wgrad_accum
        if fuse_wgrad_accum:
            _claim_main_grad(self.linear_fc1.weight, self.linear_fc2.weight)

        output = MXFP6MLPFunction.apply(
            hidden_states,
            self.linear_fc1.weight,
            self.linear_fc1.bias,
            self.linear_fc2.weight,
            fuse_wgrad_accum,
        )[0]

        # fc2 is built with skip_bias_add=True, so MLP's contract is to hand its bias back
        # unadded for the caller to fuse into a residual.
        return output, self.linear_fc2.bias
