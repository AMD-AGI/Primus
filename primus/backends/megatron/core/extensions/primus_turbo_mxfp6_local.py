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
import warnings

import torch
import torch.nn.functional as F
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLP
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    MXFP6_PROLOGUE_BIAS_GELU,
    MXFP6_PROLOGUE_BIAS_GELU_BACKWARD,
    MXFP6_PROLOGUE_IDENTITY,
    ScalingGranularity,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import (
    gemm_fp6_impl,
    gemm_fp6_out_impl,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import check_mxfp6_support

from ..models.diffusion.common.mxfp6_gates import gates
from .primus_turbo_float8_local import _quantize_fp8_tw

_GRAN_VALUE = ScalingGranularity.MX_BLOCKWISE.value

# Registered by Primus-Turbo with a pure-arithmetic fake, so it is safe to trace.
_quantize_mxfp6_dual = torch.ops.primus_turbo.quantize_mxfp6_dual_impl
_quantize_mxfp6_fused_dual = torch.ops.primus_turbo.quantize_mxfp6_fused_dual_impl
# Row-only packing, for forward passes that will never have a backward. axis=1 packs along
# the last dimension, matching the row half of the dual packer.
_quantize_mxfp6_row = torch.ops.primus_turbo.quantize_mxfp6_impl
# The QKV projection's dgrad with the QK-norm and RoPE backward folded into the prologue.
# Guarded rather than aliased unconditionally: an older Primus-Turbo has no such op, and
# MXFP6QKVNormRopeFunction is only reachable when this is present.
_quantize_mxfp6_qk_norm_rope_bwd = getattr(
    torch.ops.primus_turbo, "quantize_mxfp6_qk_norm_rope_bwd_impl", None
)
# A6W4's weight packers. Guarded like the QK-norm op above: an older Primus-Turbo has
# neither, and mxfp6_weight_format='mxfp4' is rejected at config time when they are absent.
_quantize_mxfp4_gemm_dual = getattr(torch.ops.primus_turbo, "quantize_mxfp4_gemm_dual_impl", None)
_quantize_mxfp4_gemm_row = getattr(torch.ops.primus_turbo, "quantize_mxfp4_gemm_impl", None)
# MXFP6 row + MXFP4 column from one pass. This is what makes wgrad eligible for a
# mixed-format GEMM: wgrad contracts the token dimension, so its operands are a gradient
# and an activation rather than the weight, and A6W4 cannot reach it. Packing the
# activation fp6 one way and fp4 the other lets wgrad run A6W4 with the activation as its
# narrowed B operand. Guarded like the other optional ops.
_quantize_hybrid_dual = getattr(torch.ops.primus_turbo, "quantize_mxfp6_row_mxfp4_col_dual_impl", None)


def _pack_act_dual(x, wgrad_is_fp4):
    """Pack an activation for the forward (row) and for wgrad (column).

    Under `gates().wgrad_a6w4` the column half is MXFP4, which is the operand wgrad narrows. The
    row half stays MXFP6 either way, because that is what the forward's A operand is.
    """
    if wgrad_is_fp4:
        return _quantize_hybrid_dual(x)
    return _quantize_mxfp6_dual(x)


def _pack_weight_dual(weight, weight_is_fp4):
    """Pack a weight in both contraction directions, in whichever format is configured.

    Under A6W4 this *replaces* the MXFP6 weight pack rather than adding to it, and writes
    two thirds of the bytes -- wgrad never reads the weight, so there is no third
    direction wanting the wider format.
    """
    if weight_is_fp4:
        return _quantize_mxfp4_gemm_dual(weight)
    return _quantize_mxfp6_dual(weight)


def _pack_weight_row(weight, weight_is_fp4):
    """Row direction only, for a forward with no backward behind it (eval)."""
    if weight_is_fp4:
        return _quantize_mxfp4_gemm_row(weight, 1)
    return _quantize_mxfp6_row(weight, 1)


def _wgrad_into_main_grad(weight, g_col, g_col_scale, a_col, a_col_scale, n, k, m, b_is_fp4=False):
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

    gemm_fp6_out_impl(g_col, g_col_scale, a_col, a_col_scale, main_grad, n, k, m, _GRAN_VALUE, b_is_fp4)
    return torch.empty_like(weight)


def _claim_main_grad(*weights) -> None:
    """Tell Megatron's DDP hook that these weights' gradients are already in main_grad.

    Called from the forward, which reads oddly, because the natural place -- right after
    the backward's store -- is not available: dynamo refuses to trace a mutation of state
    owned outside an autograd.Function ("HOP: Unsafe side effect"), and rather than fail it
    breaks the graph around every MXFP6 linear. On the production Flux 12B arm that fragmented
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
        weight.main_grad_initialized = True


def _reduce_grad_into_main_grad(param, partial, out_dtype, fuse_wgrad_accum, dims=0):
    """Reduce a bias-gradient partial straight into ``bias.main_grad``.

    Same trade as ``_wgrad_into_main_grad``, one operand over. Left to autograd, the
    reduction's result is handed back as ``bias.grad``, which AccumulateGrad materialises
    with a device-to-device copy before Megatron's DDP hook adds it into ``main_grad`` and
    drops it. That copy is 8.7 us of pure dispatch on a tensor of a few thousand elements,
    and there are hundreds of them per step.

    Measured on the production trace: ``__amd_rocclr_copyBuffer`` is 5.23 ms/step
    over 600 launches, 98% of them under AccumulateGrad, and the family is 94% exclusive --
    it almost never runs alongside another kernel, so what is removed here converts to step
    time nearly 1:1 rather than at the ~44% a GEMM saving converts at.

    The MXFP6 weights already avoid this and are, tellingly, absent from the copied shapes
    in that trace: ``_wgrad_into_main_grad`` writes their ``main_grad`` and returns a fresh
    placeholder, which AccumulateGrad steals instead of copying. This does the same.

    Returns the placeholder to hand back to autograd, or the ordinary gradient when the
    fusion is off or the bias has no ``main_grad`` to write.
    """
    if not (fuse_wgrad_accum and gates().fused_small_grads):
        return partial.sum(dims).to(out_dtype)
    main_grad = getattr(param, "main_grad", None)
    if main_grad is None:
        return partial.sum(dims).to(out_dtype)
    # add_ rather than copy_ because DDP zeroes the grad buffers each step and this recipe
    # runs one microbatch per optimizer step -- the same precondition _wgrad_into_main_grad
    # relies on, enforced in _init_mxfp6_linear.
    main_grad.add_(partial.sum(dims).to(main_grad.dtype))
    return torch.empty_like(param)


def _linear_mxfp6_backward(
    saved, grad_2d, m, n, k, out_dtype, orig_shape, fuse_wgrad_accum, weight_is_fp4, want_bias_grad, g_packed=None
):
    """The pure-MXFP6 half of ``MXFP6LinearFunction.backward``, lifted out verbatim so that
    ``MXFP6MLPProjFunction`` can run it on a gradient pack it shares with the MLP.
    ``g_packed`` is ``(row, row_scale, col, col_scale)`` of ``grad_2d``, or None to pack here.
    """
    if fuse_wgrad_accum:
        a_col, a_col_scale, b_col, b_col_scale, weight = saved
    else:
        a_col, a_col_scale, b_col, b_col_scale = saved

    # The bias gradient is a reduction over exactly the tensor the packer is
    # already streaming, so it rides along as a side output. Identity because
    # there is no activation to undo here, unlike the MLP's fc1.
    if g_packed is None:
        g_row, g_row_scale, g_col, g_col_scale, b_partial = _quantize_mxfp6_fused_dual(
            grad_2d, None, None, MXFP6_PROLOGUE_IDENTITY, want_bias_grad
        )
        grad_bias = b_partial.sum(0).to(out_dtype) if want_bias_grad else None
    else:
        # A caller-supplied pack of this same gradient (MXFP6MLPProjFunction). It is a plain
        # dual pack with no column sums, so it is only ever passed for an unbiased linear.
        assert not want_bias_grad, "g_packed carries no column sums; bias gradient unavailable"
        g_row, g_row_scale, g_col, g_col_scale = g_packed
        grad_bias = None

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
        out_dtype,
        _GRAN_VALUE,
        None,
        weight_is_fp4,
    )
    grad_input = grad_input.reshape(orig_shape)

    # grad_weight[N, K] = grad.T[N, M] @ input[M, K], contracting M.
    if fuse_wgrad_accum:
        grad_weight = _wgrad_into_main_grad(
            weight, g_col, g_col_scale, a_col, a_col_scale, n, k, m, gates().wgrad_a6w4
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
            out_dtype,
            _GRAN_VALUE,
            None,
            gates().wgrad_a6w4,
        )
    return grad_input, grad_weight, grad_bias


class MXFP6LinearFunction(torch.autograd.Function):
    """MXFP6 linear (Y = X @ W^T) with MX block-of-32 scaling along the contraction axis.

    Two modes via the ``backward_is_fp8`` bool primitive:

    - Pure MXFP6: forward and backward both quantize to MXFP6 and call gemm_fp6_impl.
    - Hybrid: forward is MXFP6, backward re-quantizes the saved BF16 to tensorwise FP8.

    In the pure path the forward returns the column-direction blobs as extra outputs so
    ``setup_context`` can save them; they are already uint8, so unlike MXFP4 there is no
    dtype-view juggling needed to keep the autograd engine from trying to allocate zero
    gradients in an unsupported dtype.

    The bias is an input rather than something the caller adds afterwards, so that the
    backward owns the bias gradient and can take it from the packer's column sums instead
    of paying for a separate reduction over ``grad_output``. This costs nothing in the
    forward: the biased tensor is a saved activation for the QK-norm and RoPE backward, so
    it is materialized either way, and Inductor was already doing the add as a standalone
    in-place pass over the GEMM output rather than fusing it into anything.
    """

    @staticmethod
    def forward(
        input,
        weight,
        bias,
        backward_is_fp8,
        fp8_bwd_dtype,
        fp8_gran_value,
        fp8_backend_value,
        fuse_wgrad_accum,
        grad_enabled,
        weight_is_fp4,
    ):
        out_dtype = input.dtype
        orig_shape = input.shape
        input_2d = input.reshape(-1, input.shape[-1])

        m, k = input_2d.shape
        n = weight.shape[0]

        # The column blobs are backward's operands and nothing else reads them, so a forward
        # with no backward behind it should not pay for them. Eval is entirely such a region,
        # and it is where packing hurts most, because one pack no longer amortizes across
        # fwd/dgrad/wgrad. Row-only packing is measurably cheaper than dual across the
        # Flux shapes.
        #
        # `grad_enabled` has to be sampled by the caller. Inside forward, PyTorch has already
        # cleared grad mode, so torch.is_grad_enabled() reads False even in training, and
        # ctx.needs_input_grad reads True even under no_grad; neither distinguishes the two.
        # Getting this wrong fails loudly in backward on a None operand rather than silently.
        if grad_enabled:
            a_row, a_row_scale, a_col, a_col_scale = _pack_act_dual(input_2d, gates().wgrad_a6w4)
            b_row, b_row_scale, b_col, b_col_scale = _pack_weight_dual(weight, weight_is_fp4)
        else:
            a_row, a_row_scale = _quantize_mxfp6_row(input_2d, 1)
            b_row, b_row_scale = _pack_weight_row(weight, weight_is_fp4)
            a_col = a_col_scale = b_col = b_col_scale = None

        # Bias goes into the GEMM's store epilogue, where it is free: the epilogue is bound by
        # its scatter store rather than by VALU, so the add hides completely. Handing it to the
        # GEMM rather than adding afterwards deletes a whole pass over the output, worth 7.3 ms
        # per step at the production configuration, and rounds once instead of twice.
        #
        # Passed unconditionally. Whether the installed aiter can actually fold it is Turbo's to
        # answer -- it probes, and adds the separate pass itself when it cannot -- so there is
        # nothing to gate here and no aiter version for this layer to know about.
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
            bias,
            weight_is_fp4,
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
            bias,
            backward_is_fp8,
            fp8_bwd_dtype,
            fp8_gran_value,
            fp8_backend_value,
            fuse_wgrad_accum,
            grad_enabled,
            weight_is_fp4,
        ) = inputs

        # setup_context still runs under no_grad, so this guard is load-bearing: the column
        # blobs are None there, and there is no backward to save them for anyway.
        if not grad_enabled:
            return

        ctx.backward_is_fp8 = backward_is_fp8
        ctx.fuse_wgrad_accum = fuse_wgrad_accum
        ctx.weight_is_fp4 = weight_is_fp4
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
        want_bias_grad = ctx.needs_input_grad[2]

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
            # No packer runs on this path, so the bias gradient pays for its own reduction.
            grad_bias = grad_2d.sum(0).to(ctx.out_dtype) if want_bias_grad else None
        else:
            grad_input, grad_weight, grad_bias = _linear_mxfp6_backward(
                ctx.saved_tensors,
                grad_2d,
                m,
                n,
                k,
                ctx.out_dtype,
                ctx.orig_shape,
                ctx.fuse_wgrad_accum,
                ctx.weight_is_fp4,
                want_bias_grad,
            )

        # Trailing Nones cover backward_is_fp8, fp8_bwd_dtype, fp8_gran_value,
        # fp8_backend_value, fuse_wgrad_accum, grad_enabled, weight_is_fp4.
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


def _resolve_weight_is_fp4(config) -> bool:
    """Whether this module's weight operand is MXFP4, i.e. whether it runs A6W4.

    Checked here rather than trusted from the config alone because an aiter predating
    ROCm/aiter#5587 has no `gemm_a6w4` at all, and the failure that produces is an
    AttributeError deep in the first forward. The config validation cannot see the
    installed aiter, so this is where the two meet.
    """
    if getattr(config, "mxfp6_weight_format", "mxfp6") != "mxfp4":
        return False
    from primus_turbo.pytorch.kernels.quantization.mxfp4_pack import check_a6w4_support

    supported, reason = check_a6w4_support()
    if not supported:
        raise RuntimeError(f"mxfp6_weight_format='mxfp4' is not usable here: {reason}")
    if _quantize_mxfp4_gemm_dual is None or _quantize_mxfp4_gemm_row is None:
        raise RuntimeError(
            "mxfp6_weight_format='mxfp4' needs a Primus-Turbo carrying the MXFP4 GEMM "
            "packer (quantize_mxfp4_gemm_dual_impl); this one does not."
        )
    return True


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
    module._weight_is_fp4 = _resolve_weight_is_fp4(module.config)
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

    # The bias goes through the Function rather than being added here, so that the
    # backward can lift the bias gradient out of the packer's column sums. See
    # MXFP6LinearFunction's docstring for why this is free in the forward.
    result = MXFP6LinearFunction.apply(
        input,
        weight,
        bias,
        module._backward_is_fp8,
        module._fp8_bwd_dtype,
        module._fp8_gran_value,
        module._fp8_backend_value,
        module._fuse_wgrad_accum,
        torch.is_grad_enabled(),
        module._weight_is_fp4,
    )
    return result[0]


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
# plus the bias gradient's own reduction pass in the backward. Measured on one node at
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


def _mlp_backward(
    saved, grad_output, m, k, f, h, out_dtype, orig_shape, fuse_wgrad_accum, weight_is_fp4, want_bias_grad, g2_packed=None
):
    """``MXFP6MLPFunction.backward``, lifted out verbatim so that ``MXFP6MLPProjFunction``
    can run it on a gradient pack it shares with out-proj. ``g2_packed`` is
    ``(row, row_scale, col, col_scale)`` of the fc2 output gradient, or None to pack here.
    """
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
    ) = saved

    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    g2 = grad_output.reshape(-1, h)

    if g2_packed is None:
        g2_row, g2_row_s, g2_col, g2_col_s = _quantize_mxfp6_dual(g2)
    else:
        g2_row, g2_row_s, g2_col, g2_col_s = g2_packed

    # fc2 dgrad: [m, f] = g2[m, h] @ w2[h, f], contracting h.
    grad_a = gemm_fp6_impl(
        g2_row, g2_row_s, w2_col, w2_col_s, m, f, h, out_dtype, _GRAN_VALUE, None, weight_is_fp4
    )
    # fc2 wgrad: [h, f] = g2.T[h, m] @ a[m, f], contracting m.
    if fuse_wgrad_accum:
        grad_w2 = _wgrad_into_main_grad(fused_weights[1], g2_col, g2_col_s, a_col, a_col_s, h, f, m)
    else:
        grad_w2 = gemm_fp6_impl(g2_col, g2_col_s, a_col, a_col_s, h, f, m, out_dtype, _GRAN_VALUE)

    # The GELU derivative is applied while staging, so grad_y1 is never assembled. Its
    # column sums come back as a side output because the bias gradient is a reduction
    # over exactly the tensor that no longer exists.
    g1_row, g1_row_s, g1_col, g1_col_s, b1_partial = _quantize_mxfp6_fused_dual(
        y1, grad_a, b1, MXFP6_PROLOGUE_BIAS_GELU_BACKWARD, want_bias_grad
    )

    # fc1 dgrad: [m, k] = grad_y1[m, f] @ w1[f, k], contracting f.
    grad_x = gemm_fp6_impl(
        g1_row, g1_row_s, w1_col, w1_col_s, m, k, f, out_dtype, _GRAN_VALUE, None, weight_is_fp4
    )
    grad_x = grad_x.reshape(orig_shape)
    # fc1 wgrad: [f, k] = grad_y1.T[f, m] @ x[m, k], contracting m.
    if fuse_wgrad_accum:
        grad_w1 = _wgrad_into_main_grad(
            fused_weights[0], g1_col, g1_col_s, x_col, x_col_s, f, k, m, gates().wgrad_a6w4
        )
    else:
        grad_w1 = gemm_fp6_impl(
            g1_col, g1_col_s, x_col, x_col_s, f, k, m, out_dtype, _GRAN_VALUE, None, gates().wgrad_a6w4
        )

    grad_b1 = (
        _reduce_grad_into_main_grad(b1, b1_partial, out_dtype, fuse_wgrad_accum)
        if want_bias_grad
        else None
    )

    return grad_x, grad_w1, grad_b1, grad_w2


# ---------------------------------------------------------------------------
# Grouped joint-block MLP: both streams through one GEMM per pass.
#
# A Flux joint block runs two independent streams -- image and text -- each with its own
# weights. With a stream of 8192 rows, that is 384 tiles of the shipped 256x256 macro-tile,
# which is 1.5 waves over 256 CUs and rounds up to 2, so each launch leaves about a quarter
# of the machine idle. Issuing the pair as one 2-group GEMM (16384 rows = 3 exact waves)
# recovers that. The grouped kernel picks a different B per group, which is what makes this
# legal for streams that share no weights.
#
# Which blobs can be shared is not uniform, and getting it wrong is silent:
#   * activation ROW blobs -> shared. They are the grouped A operand.
#   * activation COLUMN blobs -> per-stream. wgrad contracts M, and the two streams have
#     different weights, so a stacked column pack would sum contributions belonging to
#     different matrices.
#   * weight ROW and COLUMN blobs -> both shared. The column blob is only ever the B
#     operand of a grouped dgrad, never a wgrad operand.
#
# Measured per pair at the production shape (m=8192, k=3072, f=12288, h=3072), packing
# included, against two separate streams:
#     fwd fc1  (N=12288)   499.9 -> 463.6 us
#     fwd fc2  (N= 3072)   514.6 -> 477.8 us
#     bwd fc2 dgrad        457.5 -> 458.7 us
#     bwd fc1 dgrad        510.9 -> 475.3 us
#                          total -107.5 us/pair -> -2.04 ms/step over 19 joint blocks
# Grouping every GEMM beats grouping only the N=3072 ones (-1.37 ms), which is not what a
# pure wave-quantisation model predicts: the out= packers also drop two allocations per
# pair, and that part does not depend on N.
#
# Treat that as an upper bound. Three times now, measurement has shown a GEMM-level gain
# shrink once the surrounding plumbing was timed, and the fused prologue still runs once
# per stream here because the two streams have different fc1 biases.
#
# Off by default. Enable with `mxfp6_grouped_mlp: true`.
# ---------------------------------------------------------------------------

# The 2-group A6W6 kernel, built with the same flags as the variant production already
# runs for these shapes -- LDSTAGE=1 STNT=1 SWZTH=512 -- so grouping is not paying for a
# kernel-variant downgrade at the same time. aiter's host gate admits an oversized B only
# for kernels whose name carries "_wgrp" (weight-grouped), so every other kernel keeps its
# strict size equality; "_grp" would also have matched dmabig_grp16/grp64, which are
# ordinary single-weight kernels that carry a rasterization group size in their names.
_GRP_KERNEL_NAME = "f6gemm_a6w6_stnt_allk_wgrp2_kernel_func"

# Caller-buffer packers. Guarded like the other optional Turbo ops: an older Primus-Turbo
# has none of them, and the gate above is rejected at import time when they are absent.
# Note these live in the `primus_turbo_cpp_extension` namespace rather than `primus_turbo`
# like their allocating siblings -- an inconsistency in Turbo, not a choice here.
_TURBO_CPP = getattr(torch.ops, "primus_turbo_cpp_extension", None)
_quantize_mxfp6_dual_out = getattr(_TURBO_CPP, "quantize_mxfp6_dual_out", None)
_quantize_mxfp6_fused_dual_out = getattr(_TURBO_CPP, "quantize_mxfp6_fused_dual_out", None)


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
    def forward(hidden_states, w1, b1, w2, fuse_wgrad_accum, grad_enabled, weight_is_fp4):
        out_dtype = hidden_states.dtype
        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, orig_shape[-1])

        m, k = x.shape
        f = w1.shape[0]
        h = w2.shape[0]

        # Same reasoning as MXFP6LinearFunction: the column blobs are backward's operands,
        # so a no-grad forward should not pay for them. See the note there on why
        # grad_enabled has to be sampled by the caller rather than read here.
        if grad_enabled:
            x_row, x_row_s, x_col, x_col_s = _pack_act_dual(x, gates().wgrad_a6w4)
            w1_row, w1_row_s, w1_col, w1_col_s = _pack_weight_dual(w1, weight_is_fp4)
        else:
            x_row, x_row_s = _quantize_mxfp6_row(x, 1)
            w1_row, w1_row_s = _pack_weight_row(w1, weight_is_fp4)
            x_col = x_col_s = w1_col = w1_col_s = None

        # Pre-activation. Saved for backward, where the epilogue is recomputed from it
        # rather than its output being stashed -- the same bytes are held either way.
        y1 = gemm_fp6_impl(
            x_row, x_row_s, w1_row, w1_row_s, m, f, k, out_dtype, _GRAN_VALUE, None, weight_is_fp4
        )

        # gelu(y1 + b1), packed in both directions without ever being written out.
        if grad_enabled:
            a_row, a_row_s, a_col, a_col_s, _ = _quantize_mxfp6_fused_dual(
                y1, None, b1, MXFP6_PROLOGUE_BIAS_GELU, False
            )
            w2_row, w2_row_s, w2_col, w2_col_s = _pack_weight_dual(w2, weight_is_fp4)
        else:
            # The fused packer has no row-only mode, so the activation still costs a dual
            # pass; only its GELU epilogue matters here and that is shared.
            a_row, a_row_s, a_col, a_col_s, _ = _quantize_mxfp6_fused_dual(
                y1, None, b1, MXFP6_PROLOGUE_BIAS_GELU, False
            )
            w2_row, w2_row_s = _pack_weight_row(w2, weight_is_fp4)
            w2_col = w2_col_s = None

        output = gemm_fp6_impl(
            a_row, a_row_s, w2_row, w2_row_s, m, h, f, out_dtype, _GRAN_VALUE, None, weight_is_fp4
        )
        output = output.reshape(*orig_shape[:-1], h)

        return output, y1, x_col, x_col_s, a_col, a_col_s, w1_col, w1_col_s, w2_col, w2_col_s

    @staticmethod
    def setup_context(ctx, inputs, output):
        hidden_states, w1, b1, w2, fuse_wgrad_accum, grad_enabled, weight_is_fp4 = inputs

        # setup_context still runs under no_grad, where the column blobs are None.
        if not grad_enabled:
            return

        ctx.fuse_wgrad_accum = fuse_wgrad_accum
        ctx.out_dtype = hidden_states.dtype
        ctx.orig_shape = hidden_states.shape
        # The packed blobs carry no shape, so the logical dims have to be saved too.
        ctx.m = hidden_states.numel() // hidden_states.shape[-1]
        ctx.k = hidden_states.shape[-1]
        ctx.f = w1.shape[0]
        ctx.h = w2.shape[0]
        ctx.weight_is_fp4 = weight_is_fp4

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
        grad_x, grad_w1, grad_b1, grad_w2 = _mlp_backward(
            ctx.saved_tensors,
            grad_output,
            ctx.m,
            ctx.k,
            ctx.f,
            ctx.h,
            ctx.out_dtype,
            ctx.orig_shape,
            ctx.fuse_wgrad_accum,
            ctx.weight_is_fp4,
            ctx.needs_input_grad[2],
        )
        # Trailing Nones cover fuse_wgrad_accum, grad_enabled and weight_is_fp4.
        return grad_x, grad_w1, grad_b1, grad_w2, None, None, None


def _grouped_mlp_unavailable_reason():
    """Why the grouped MLP cannot run here, or None if it can."""
    if _quantize_mxfp6_dual_out is None or _quantize_mxfp6_fused_dual_out is None:
        return "this Primus-Turbo has no quantize_mxfp6_*_out ops"
    if gates().wgrad_a6w4:
        return "PRIMUS_MXFP6gates().wgrad_a6w4 packs the column half as MXFP4; not wired here"
    return None


def _pair_buffers(rows, k, device):
    """A buffer pair sized for two stacked ``rows x k`` packs, plus the single-pack stride."""
    from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import mxfp6_pack_sizes

    pn, sn = mxfp6_pack_sizes(rows, k)
    return (
        torch.empty(2 * pn, dtype=torch.uint8, device=device),
        torch.empty(2 * sn, dtype=torch.uint8, device=device),
        pn,
        sn,
    )


def _pack_pair_act(xs):
    """Row blobs into halves of one buffer; column blobs per stream."""
    from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import mxfp6_pack_sizes

    m, k = xs[0].shape
    dev = xs[0].device
    row_p, row_s, pn, sn = _pair_buffers(m, k, dev)
    cpn, csn = mxfp6_pack_sizes(k, m)
    cols = []
    for i, x in enumerate(xs):
        cp = torch.empty(cpn, dtype=torch.uint8, device=dev)
        cs = torch.empty(csn, dtype=torch.uint8, device=dev)
        _quantize_mxfp6_dual_out(x, row_p[i * pn : (i + 1) * pn], row_s[i * sn : (i + 1) * sn], cp, cs)
        cols.append((cp, cs))
    return row_p, row_s, cols


def _pack_pair_weight(ws):
    """Both directions into halves of shared buffers -- neither is a wgrad operand."""

    n, k = ws[0].shape
    dev = ws[0].device
    row_p, row_s, pn, sn = _pair_buffers(n, k, dev)
    col_p, col_s, cpn, csn = _pair_buffers(k, n, dev)
    for i, w in enumerate(ws):
        _quantize_mxfp6_dual_out(
            w,
            row_p[i * pn : (i + 1) * pn],
            row_s[i * sn : (i + 1) * sn],
            col_p[i * cpn : (i + 1) * cpn],
            col_s[i * csn : (i + 1) * csn],
        )
    return row_p, row_s, col_p, col_s


def _fused_prologue_pair(ys, aux, biases, mode, want_col_sum):
    """Run the fused prologue once per stream, row blobs landing in one shared buffer.

    Once per stream rather than once for the pair because the prologue applies a per-column
    bias and the two streams have different ones. Only the destination is shared.
    """
    from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import (
        mxfp6_col_sum_rows,
        mxfp6_pack_sizes,
    )

    m, n = ys[0].shape
    dev = ys[0].device
    row_p, row_s, pn, sn = _pair_buffers(m, n, dev)
    cpn, csn = mxfp6_pack_sizes(n, m)
    cols, partials = [], []
    for i, y in enumerate(ys):
        cp = torch.empty(cpn, dtype=torch.uint8, device=dev)
        cs = torch.empty(csn, dtype=torch.uint8, device=dev)
        part = (
            torch.empty(mxfp6_col_sum_rows(m), n, dtype=torch.float32, device=dev) if want_col_sum else None
        )
        _quantize_mxfp6_fused_dual_out(
            y,
            aux[i] if aux is not None else None,
            biases[i],
            mode,
            row_p[i * pn : (i + 1) * pn],
            row_s[i * sn : (i + 1) * sn],
            cp,
            cs,
            part,
        )
        cols.append((cp, cs))
        partials.append(part)
    return row_p, row_s, cols, partials


def _grouped_gemm_raw(a, a_s, b, b_s, m_total, n, k):
    """One A6W6 GEMM over two groups stacked along M, B carrying both streams' weights.

    Calls aiter directly rather than ``gemm_fp6_impl``: that entry point has no kernel-name
    argument, and its ``_validate_blobs`` asserts B is exactly one weight, which is the very
    check the grouped host gate relaxes.
    """
    import aiter

    return aiter.gemm_a6w6(a, b, a_s, b_s, m_total, n, k, kernelName=_GRP_KERNEL_NAME)


@torch.library.custom_op("primus::mxfp6_grouped_gemm", mutates_args=(), device_types="cuda")
def _grouped_gemm_functional(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor, m_total: int, n: int, k: int
) -> torch.Tensor:
    """``_grouped_gemm_raw`` behind a functional schema.

    aiter registers its ctypes GEMM (``aiter._gemm_a6w6_asm``) with ``mutates_args="unknown"``,
    which torch.compile must read as "may write every tensor argument". Functionalization
    therefore clones each operand that is still live afterwards -- for the grouped backward
    that is the saved weight column pair and its scales, which Inductor then copies onto
    themselves in place: four ``as_strided_clone`` kernels per joint block of
    identity copies. This op declares the truth (it only writes the tensor it returns), so
    nothing is cloned. Turbo's ``gemm_fp6_impl`` already does the same for every other
    MXFP6 GEMM, which is why only the grouped path showed the copies.
    """
    out = _grouped_gemm_raw(a, a_s, b, b_s, m_total, n, k)
    # A custom op may not return a view; gemm_a6w6 returns one only when it had to pad,
    # which the grouped shapes (M, N multiples of 256) never need.
    return out if out._base is None else out.contiguous()


@_grouped_gemm_functional.register_fake
def _(a, a_s, b, b_s, m_total, n, k):
    return torch.empty((m_total, n), dtype=torch.bfloat16, device=a.device)


def _grouped_gemm(a, a_s, b, b_s, m_total, n, k):
    if gates().grouped_gemm_functional:
        return _grouped_gemm_functional(a, a_s, b, b_s, m_total, n, k)
    return _grouped_gemm_raw(a, a_s, b, b_s, m_total, n, k)


class MXFP6GroupedMLPFunction(torch.autograd.Function):
    """Both joint-block streams' MLPs, one grouped GEMM per pass.

    Mirrors ``MXFP6MLPFunction`` operand for operand; the only difference is that every
    GEMM carries two groups and the packers write into shared buffers. The activation is
    still never written to HBM, and the GELU derivative is still applied while staging, so
    ``grad_y1`` is never assembled and the bias gradient still returns as column sums.
    """

    @staticmethod
    def forward(x_a, x_b, w1_a, w1_b, b1_a, b1_b, w2_a, w2_b, fuse_wgrad_accum, grad_enabled, weight_is_fp4):
        x_a.dtype
        orig_shape = x_a.shape
        # No .contiguous() here: MXFP6MLPFunction does not call it either, and adding it
        # made Inductor materialise the reshape -- 38 as_strided_clone launches per step.
        xa = x_a.reshape(-1, orig_shape[-1])
        xb = x_b.reshape(-1, orig_shape[-1])
        m, k = xa.shape
        f = w1_a.shape[0]
        h = w2_a.shape[0]

        x_row, x_row_s, x_cols = _pack_pair_act((xa, xb))
        w1_row, w1_row_s, w1_col, w1_col_s = _pack_pair_weight((w1_a, w1_b))

        y1 = _grouped_gemm(x_row, x_row_s, w1_row, w1_row_s, 2 * m, f, k)

        a_row, a_row_s, a_cols, _ = _fused_prologue_pair(
            (y1[:m], y1[m:]), None, (b1_a, b1_b), MXFP6_PROLOGUE_BIAS_GELU, False
        )
        w2_row, w2_row_s, w2_col, w2_col_s = _pack_pair_weight((w2_a, w2_b))

        out = _grouped_gemm(a_row, a_row_s, w2_row, w2_row_s, 2 * m, h, f)
        out_a = out[:m].reshape(*orig_shape[:-1], h)
        out_b = out[m:].reshape(*orig_shape[:-1], h)

        return (
            out_a,
            out_b,
            y1,
            x_cols[0][0],
            x_cols[0][1],
            x_cols[1][0],
            x_cols[1][1],
            a_cols[0][0],
            a_cols[0][1],
            a_cols[1][0],
            a_cols[1][1],
            w1_col,
            w1_col_s,
            w2_col,
            w2_col_s,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x_a, x_b, w1_a, w1_b, b1_a, b1_b, w2_a, w2_b, fuse_wgrad_accum, grad_enabled, weight_is_fp4) = inputs
        if not grad_enabled:
            return
        ctx.fuse_wgrad_accum = fuse_wgrad_accum
        ctx.out_dtype = x_a.dtype
        ctx.orig_shape = x_a.shape
        ctx.m = x_a.numel() // x_a.shape[-1]
        ctx.k = x_a.shape[-1]
        ctx.f = w1_a.shape[0]
        ctx.h = w2_a.shape[0]
        ctx.weight_is_fp4 = weight_is_fp4
        blobs = output[3:]
        extra = (w1_a, w1_b, w2_a, w2_b) if fuse_wgrad_accum else ()
        ctx.save_for_backward(output[2], b1_a, b1_b, *blobs, *extra)
        ctx.mark_non_differentiable(*blobs)

    @staticmethod
    def backward(ctx, g_out_a, g_out_b, *_):
        (
            y1,
            b1_a,
            b1_b,
            xc_a,
            xcs_a,
            xc_b,
            xcs_b,
            ac_a,
            acs_a,
            ac_b,
            acs_b,
            w1_col,
            w1_col_s,
            w2_col,
            w2_col_s,
            *fused,
        ) = ctx.saved_tensors
        m, k, f, h = ctx.m, ctx.k, ctx.f, ctx.h
        out_dtype = ctx.out_dtype

        if not g_out_a.is_contiguous():
            g_out_a = g_out_a.contiguous()
        if not g_out_b.is_contiguous():
            g_out_b = g_out_b.contiguous()
        g2a = g_out_a.reshape(-1, h)
        g2b = g_out_b.reshape(-1, h)
        g2_row, g2_row_s, g2_cols = _pack_pair_act((g2a, g2b))

        # fc2 dgrad, both streams: [2m, f] = g2[2m, h] @ w2[h, f], contracting h.
        grad_a = _grouped_gemm(g2_row, g2_row_s, w2_col, w2_col_s, 2 * m, f, h)

        # fc2 wgrad stays per stream: it contracts m, so its operands belong to one stream.
        if ctx.fuse_wgrad_accum:
            grad_w2_a = _wgrad_into_main_grad(fused[2], g2_cols[0][0], g2_cols[0][1], ac_a, acs_a, h, f, m)
            grad_w2_b = _wgrad_into_main_grad(fused[3], g2_cols[1][0], g2_cols[1][1], ac_b, acs_b, h, f, m)
        else:
            grad_w2_a = gemm_fp6_impl(
                g2_cols[0][0], g2_cols[0][1], ac_a, acs_a, h, f, m, out_dtype, _GRAN_VALUE
            )
            grad_w2_b = gemm_fp6_impl(
                g2_cols[1][0], g2_cols[1][1], ac_b, acs_b, h, f, m, out_dtype, _GRAN_VALUE
            )

        want_bias_grad = ctx.needs_input_grad[4]
        g1_row, g1_row_s, g1_cols, partials = _fused_prologue_pair(
            (y1[:m], y1[m:]),
            (grad_a[:m], grad_a[m:]),
            (b1_a, b1_b),
            MXFP6_PROLOGUE_BIAS_GELU_BACKWARD,
            want_bias_grad,
        )

        # fc1 dgrad, both streams: [2m, k] = grad_y1[2m, f] @ w1[f, k], contracting f.
        grad_x = _grouped_gemm(g1_row, g1_row_s, w1_col, w1_col_s, 2 * m, k, f)
        grad_x_a = grad_x[:m].reshape(ctx.orig_shape)
        grad_x_b = grad_x[m:].reshape(ctx.orig_shape)

        if ctx.fuse_wgrad_accum:
            grad_w1_a = _wgrad_into_main_grad(
                fused[0], g1_cols[0][0], g1_cols[0][1], xc_a, xcs_a, f, k, m, gates().wgrad_a6w4
            )
            grad_w1_b = _wgrad_into_main_grad(
                fused[1], g1_cols[1][0], g1_cols[1][1], xc_b, xcs_b, f, k, m, gates().wgrad_a6w4
            )
        else:
            grad_w1_a = gemm_fp6_impl(
                g1_cols[0][0],
                g1_cols[0][1],
                xc_a,
                xcs_a,
                f,
                k,
                m,
                out_dtype,
                _GRAN_VALUE,
                None,
                gates().wgrad_a6w4,
            )
            grad_w1_b = gemm_fp6_impl(
                g1_cols[1][0],
                g1_cols[1][1],
                xc_b,
                xcs_b,
                f,
                k,
                m,
                out_dtype,
                _GRAN_VALUE,
                None,
                gates().wgrad_a6w4,
            )

        if want_bias_grad:
            grad_b1_a = _reduce_grad_into_main_grad(b1_a, partials[0], out_dtype, ctx.fuse_wgrad_accum)
            grad_b1_b = _reduce_grad_into_main_grad(b1_b, partials[1], out_dtype, ctx.fuse_wgrad_accum)
        else:
            grad_b1_a = grad_b1_b = None

        # Trailing Nones cover fuse_wgrad_accum, grad_enabled and weight_is_fp4.
        return (
            grad_x_a,
            grad_x_b,
            grad_w1_a,
            grad_w1_b,
            grad_b1_a,
            grad_b1_b,
            grad_w2_a,
            grad_w2_b,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Whole-QKV fusion: linear_qkv -> QK-norm -> RoPE.
#
# The same argument as the MLP above, one region over. Splitting the projection from the
# norm+RoPE forces d(mixed_qkv) to exist: the norm's backward has to write a real tensor and
# the projection's backward has to read one. Today that write is a Triton kernel's output in
# bf16 and the read is the packer picking it straight back up. Owning
# linear_qkv -> norm -> rope in one Function lets the packer compute the gradient itself while
# staging the tile it is about to pack, from the nine operands the norm and rotation already
# saved.
#
# Unlike the MLP fusion this is *not* an elementwise prologue. It computes
#   dx = rstd * (dn * w - u_hat * mean_d(dn * w * u_hat))
# where dn is the rotation's backward, and that mean is a reduction over head_dim -- which is
# why the packer runs at a tile width of head_dim for this prologue, and why num_heads has to
# be even and head_dim exactly 128. It also emits the two norm weights' gradients as side
# outputs, for the same reason the MLP's bias gradient comes back as column sums: the tensor
# they would be reduced from no longer exists.
#
# Priced on the shapes it runs at rather than modelled: the prologue costs roughly half of
# the kernel time it removes, and most of that difference survives into wall clock. The
# measured account -- the figures, a probe that predicted a lower cost, and why production
# disagrees with it -- is in the campaign's packer/RESULTS_qkr_kernel.md.
#
# Correctness is gated two ways. packer/qkr_exact_test.cu proves the packed blobs are
# byte-identical to packing a host-computed dx; packer/qkr_triton_gate.py proves that dx is
# the function Flux's Triton backward computes, to 3.2e-4 on d(mixed_qkv) (bf16's own
# resolution) and 2e-7 on the weight gradients. Packing the fused output against packing
# Triton's materialised d_qkv differs in 5 bytes of 10.6 million with every scale identical.
# ---------------------------------------------------------------------------


class MXFP6QKVNormRopeFunction(torch.autograd.Function):
    """QKV projection, QK-norm and RoPE as one op, with d(mixed_qkv) never in HBM.

    Follows ``MXFP6MLPFunction``'s conventions: the column-direction blobs and the norm's
    saved state leave as extra outputs so ``setup_context`` can save them and mark them
    non-differentiable, and all non-tensor state lands on ``ctx`` as primitives so
    ``torch.compile`` traces cleanly.

    ``mixed_qkv`` is still saved, but it was already a saved activation for the norm and
    rotation's own backward, so this holds no more bytes than the unfused path -- the same
    trade the MLP makes with its pre-activation.

    The rotary embedding **must be interleaved**. Nothing in the signature makes that visible
    and no check will catch a half-split caller; ``_fused_qkv_unusable_reason`` tests the
    config field, which is the only place it is knowable.
    """

    @staticmethod
    def forward(
        hidden_states,
        w_qkv,
        b_qkv,
        wq,
        wk,
        cos,
        sin,
        eps,
        interleaved,
        fuse_wgrad_accum,
        grad_enabled,
        weight_is_fp4,
    ):
        from primus.backends.megatron.core.models.diffusion.common.fused_norm_rope import (
            _qkv_fwd,
            _qkv_fwd_nov,
        )

        out_dtype = hidden_states.dtype
        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, orig_shape[-1])

        m, k = x.shape
        n = w_qkv.shape[0]  # num_heads * 3 * head_dim
        d = wq.shape[0]
        h = n // (3 * d)

        # Same reasoning as MXFP6LinearFunction: the column blobs are backward's operands, so
        # a no-grad forward should not pay for them.
        if grad_enabled:
            x_row, x_row_s, x_col, x_col_s = _pack_act_dual(x, gates().wgrad_a6w4)
            w_row, w_row_s, w_col, w_col_s = _pack_weight_dual(w_qkv, weight_is_fp4)
        else:
            x_row, x_row_s = _quantize_mxfp6_row(x, 1)
            w_row, w_row_s = _pack_weight_row(w_qkv, weight_is_fp4)
            x_col = x_col_s = w_col = w_col_s = None

        # The bias is folded into the GEMM's epilogue, so mixed_qkv is the biased projection
        # -- which is what the norm consumes and what the prologue re-reads in the backward.
        mixed_qkv = gemm_fp6_impl(
            x_row, x_row_s, w_row, w_row_s, m, n, k, out_dtype, _GRAN_VALUE, b_qkv, weight_is_fp4
        )

        # The norm and rotation, unchanged. This is the production Triton op, on the
        # [..., num_heads, 3 * head_dim] view it expects; only its *backward* is replaced.
        qkv = mixed_qkv.reshape(*orig_shape[:-1], h, 3 * d)
        if gates().strided_v:
            # Skip the V repack and hand FMHA the strided slice. Safe *here* specifically:
            # this runs inside autograd.Function.forward, where grad mode is off, so the
            # slice is not tracked and no second autograd path is created. Taking the same
            # slice on the eager path (fused_norm_rope.py:596, under register_autograd)
            # would add one, whose backward allocates a full [S,B,H,3D] zero buffer and
            # scatters dv into it -- about 3x the traffic the repack costs.
            #
            # qkv is a view of mixed_qkv, which is produced inside this forward rather than
            # passed in, so returning a view of it is not a view-of-an-input.
            q, k_out, q_rstd, k_rstd = _qkv_fwd_nov(qkv, wq, wk, cos, sin, eps, interleaved)
            v = qkv[..., 2 * d :]
        else:
            q, k_out, v, q_rstd, k_rstd = _qkv_fwd(qkv, wq, wk, cos, sin, eps, interleaved)

        return q, k_out, v, mixed_qkv, q_rstd, k_rstd, x_col, x_col_s, w_col, w_col_s

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            hidden_states,
            w_qkv,
            b_qkv,
            wq,
            wk,
            cos,
            sin,
            eps,
            interleaved,
            fuse_wgrad_accum,
            grad_enabled,
            weight_is_fp4,
        ) = inputs

        # setup_context still runs under no_grad, where the column blobs are None.
        if not grad_enabled:
            return

        ctx.fuse_wgrad_accum = fuse_wgrad_accum
        ctx.out_dtype = hidden_states.dtype
        ctx.orig_shape = hidden_states.shape
        # The packed blobs carry no shape, so the logical dims have to be saved too.
        ctx.m = hidden_states.numel() // hidden_states.shape[-1]
        ctx.k = hidden_states.shape[-1]
        ctx.n = w_qkv.shape[0]
        ctx.d = wq.shape[0]
        ctx.h = ctx.n // (3 * ctx.d)
        ctx.weight_is_fp4 = weight_is_fp4

        _, _, _, mixed_qkv, q_rstd, k_rstd, x_col, x_col_s, w_col, w_col_s = output
        blobs = (x_col, x_col_s, w_col, w_col_s)
        # wq/wk are leaf parameters and cos/sin are built once per step, so saving them costs
        # nothing; the prologue needs all four, plus the rstd the forward just computed.
        extra = (w_qkv,) if fuse_wgrad_accum else ()
        ctx.save_for_backward(mixed_qkv, wq, wk, cos, sin, q_rstd, k_rstd, *blobs, *extra)
        ctx.mark_non_differentiable(mixed_qkv, q_rstd, k_rstd, *blobs)

    @staticmethod
    def backward(ctx, dq, dk, dv, *_):
        (
            mixed_qkv,
            wq,
            wk,
            cos,
            sin,
            q_rstd,
            k_rstd,
            x_col,
            x_col_s,
            w_col,
            w_col_s,
            *fused_weights,
        ) = ctx.saved_tensors
        m, k, n, h, d = ctx.m, ctx.k, ctx.n, ctx.h, ctx.d
        out_dtype = ctx.out_dtype

        # The prologue reads all three slices as [m, num_heads * head_dim]. dv is included
        # because v is still packed -- plainly, but packed -- and that read is the
        # d_qkv[..., 2D:].copy_(dv) the fusion absorbs.
        grads = tuple(g.contiguous().reshape(m, h * d) for g in (dq, dk, dv))

        want_bias_grad = ctx.needs_input_grad[2]
        (
            g_row,
            g_row_s,
            g_col,
            g_col_s,
            b_partial,
            dwq_partial,
            dwk_partial,
        ) = _quantize_mxfp6_qk_norm_rope_bwd(
            mixed_qkv, *grads, cos, sin, wq, wk, q_rstd, k_rstd, want_bias_grad
        )

        # dgrad: [m, k] = d(mixed_qkv)[m, n] @ w_qkv[n, k], contracting n.
        grad_x = gemm_fp6_impl(
            g_row, g_row_s, w_col, w_col_s, m, k, n, out_dtype, _GRAN_VALUE, None, ctx.weight_is_fp4
        )
        grad_x = grad_x.reshape(ctx.orig_shape)
        # wgrad: [n, k] = d(mixed_qkv).T[n, m] @ x[m, k], contracting m.
        if ctx.fuse_wgrad_accum:
            grad_w = _wgrad_into_main_grad(
                fused_weights[0], g_col, g_col_s, x_col, x_col_s, n, k, m, gates().wgrad_a6w4
            )
        else:
            grad_w = gemm_fp6_impl(
                g_col, g_col_s, x_col, x_col_s, n, k, m, out_dtype, _GRAN_VALUE, None, gates().wgrad_a6w4
            )

        # The QKV bias is not in saved_tensors, so it cannot be routed into main_grad from
        # here without widening the save set. Its [9216] copies are left on the table.
        grad_b = b_partial.sum(0).to(out_dtype) if want_bias_grad else None
        # dw reduces over rows *and* heads: the norm weight is [head_dim] and shared across
        # heads, and a packer block owns one head, so the partial buffer carries both axes.
        grad_wq = _reduce_grad_into_main_grad(wq, dwq_partial, wq.dtype, ctx.fuse_wgrad_accum, dims=(0, 1))
        grad_wk = _reduce_grad_into_main_grad(wk, dwk_partial, wk.dtype, ctx.fuse_wgrad_accum, dims=(0, 1))

        # Trailing Nones cover cos, sin, eps, interleaved, fuse_wgrad_accum, grad_enabled
        # and weight_is_fp4.
        return grad_x, grad_w, grad_b, grad_wq, grad_wk, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Joint QKV: both streams' projections and norm+RoPE, producing joint q/k/v directly.
#
# Flux's joint blocks project the two streams separately and then join them with three
# torch.cat calls before attention. The reference does the same thing
# (torchtitan/.../flux/model/layers.py: `q = torch.cat((txt_q, img_q), dim=2)`), so the
# concatenation is part of the model -- but *performing* it as a separate pass is not. At
# the production configuration, those cats and their backward splits are ~3.1 ms/step of pure data movement.
#
# Writing both streams into one buffer removes them:
#   * mixed_qkv is one [m_a + m_b, n] tensor, each projection writing its own half through
#     gemm_fp6_out_impl. V then falls out as a single strided slice spanning both streams,
#     so the V cat disappears with no extra work.
#   * q and k are joint tensors, and each stream's norm+RoPE writes into its own slice via
#     primus::fused_qk_norm_rope_into. The two streams keep their own QK-norm weights and
#     their own RoPE tables, exactly as the reference has them -- only the destination is
#     shared.
#
# Arithmetic is unchanged operand for operand; this is a layout change, not an
# approximation. Off by default: PRIMUS_MXFP6_JOINT_QKV=1.
#
# Requires PRIMUS_MXFP6_STRIDED_V: V is taken as a slice of the joint mixed_qkv, which is
# the same trick that gate already relies on, and the non-strided path would need a joint
# repack destination that does not exist.
#
# Not grouped. The two projections stay two GEMMs writing into halves rather than one
# 2-group GEMM. Grouping them is worth a further ~0.15 ms of forward GEMM time and needs
# the operands packed into shared buffers; the cat removal is ~90% of the item and does not
# depend on it. The backward stays per stream for a harder reason: its packer prologue
# (quantize_mxfp6_qk_norm_rope_bwd) has no caller-buffer variant, so the two streams'
# packed gradients cannot land adjacently for a grouped dgrad.
# ---------------------------------------------------------------------------


def joint_qkv_enabled() -> bool:
    """Is the joint QKV path switched on and usable in this configuration?"""
    g = gates()
    return g.joint_qkv and g.strided_v and _quantize_mxfp6_qk_norm_rope_bwd is not None


class MXFP6JointQKVFunction(torch.autograd.Function):
    """Both joint-block streams' QKV projection + QK-norm + RoPE, emitting joint q/k/v.

    Stream ``a`` is the one that leads the joint sequence (the context/text stream, which
    ``torch.cat([added, main])`` used to put first); stream ``b`` follows it.

    Mirrors ``MXFP6QKVNormRopeFunction`` operand for operand. The backward runs per stream
    on slices of the joint gradients -- each stream owns its norm weights, RoPE table and
    saved rstd, so the fused prologue has to see them separately anyway.
    """

    @staticmethod
    def forward(
        x_a,
        x_b,
        w_a,
        w_b,
        b_a,
        b_b,
        wq_a,
        wk_a,
        wq_b,
        wk_b,
        cos_a,
        sin_a,
        cos_b,
        sin_b,
        eps,
        interleaved,
        fuse_wgrad_accum,
        grad_enabled,
        weight_is_fp4,
    ):
        from primus.backends.megatron.core.models.diffusion.common.fused_norm_rope import (
            _qkv_fwd_into,
        )

        out_dtype = x_a.dtype
        shape_a, shape_b = x_a.shape, x_b.shape
        xa = x_a.reshape(-1, shape_a[-1])
        xb = x_b.reshape(-1, shape_b[-1])
        m_a, k = xa.shape
        m_b = xb.shape[0]
        n = w_a.shape[0]
        d = wq_a.shape[0]
        h = n // (3 * d)

        packs = []
        for x, w in ((xa, w_a), (xb, w_b)):
            if grad_enabled:
                x_row, x_row_s, x_col, x_col_s = _pack_act_dual(x, gates().wgrad_a6w4)
                w_row, w_row_s, w_col, w_col_s = _pack_weight_dual(w, weight_is_fp4)
            else:
                x_row, x_row_s = _quantize_mxfp6_row(x, 1)
                w_row, w_row_s = _pack_weight_row(w, weight_is_fp4)
                x_col = x_col_s = w_col = w_col_s = None
            packs.append((x_row, x_row_s, x_col, x_col_s, w_row, w_row_s, w_col, w_col_s))

        # One buffer for both projections. Rows are sequence-major within a stream, so
        # stacking stream a above stream b and reshaping gives exactly the joint sequence
        # the cat used to build.
        mixed_qkv = torch.empty(m_a + m_b, n, device=xa.device, dtype=out_dtype)
        gemm_fp6_out_impl(
            packs[0][0],
            packs[0][1],
            packs[0][4],
            packs[0][5],
            mixed_qkv[:m_a],
            m_a,
            n,
            k,
            _GRAN_VALUE,
            weight_is_fp4,
        )
        gemm_fp6_out_impl(
            packs[1][0],
            packs[1][1],
            packs[1][4],
            packs[1][5],
            mixed_qkv[m_a:],
            m_b,
            n,
            k,
            _GRAN_VALUE,
            weight_is_fp4,
        )

        s_a, s_b, batch = shape_a[0], shape_b[0], shape_a[1]
        qkv = mixed_qkv.reshape(s_a + s_b, batch, h, 3 * d)
        q = torch.empty(s_a + s_b, batch, h, d, device=xa.device, dtype=out_dtype)
        k_out = torch.empty_like(q)
        q_rstd_a, k_rstd_a = _qkv_fwd_into(
            qkv[:s_a], wq_a, wk_a, cos_a, sin_a, eps, interleaved, q[:s_a], k_out[:s_a]
        )
        q_rstd_b, k_rstd_b = _qkv_fwd_into(
            qkv[s_a:], wq_b, wk_b, cos_b, sin_b, eps, interleaved, q[s_a:], k_out[s_a:]
        )
        # One strided slice covering both streams -- the V cat falls out for free.
        v = qkv[..., 2 * d :]

        return (
            q,
            k_out,
            v,
            mixed_qkv,
            q_rstd_a,
            k_rstd_a,
            q_rstd_b,
            k_rstd_b,
            packs[0][2],
            packs[0][3],
            packs[0][6],
            packs[0][7],
            packs[1][2],
            packs[1][3],
            packs[1][6],
            packs[1][7],
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            x_a,
            x_b,
            w_a,
            w_b,
            b_a,
            b_b,
            wq_a,
            wk_a,
            wq_b,
            wk_b,
            cos_a,
            sin_a,
            cos_b,
            sin_b,
            eps,
            interleaved,
            fuse_wgrad_accum,
            grad_enabled,
            weight_is_fp4,
        ) = inputs
        if not grad_enabled:
            return
        ctx.fuse_wgrad_accum = fuse_wgrad_accum
        ctx.out_dtype = x_a.dtype
        ctx.shape_a, ctx.shape_b = x_a.shape, x_b.shape
        ctx.m_a = x_a.numel() // x_a.shape[-1]
        ctx.m_b = x_b.numel() // x_b.shape[-1]
        ctx.s_a = x_a.shape[0]
        ctx.k = x_a.shape[-1]
        ctx.n = w_a.shape[0]
        ctx.d = wq_a.shape[0]
        ctx.h = ctx.n // (3 * ctx.d)
        ctx.eps, ctx.interleaved = eps, interleaved
        ctx.weight_is_fp4 = weight_is_fp4
        blobs = output[3:]
        extra = (w_a, w_b) if fuse_wgrad_accum else ()
        ctx.save_for_backward(wq_a, wk_a, wq_b, wk_b, cos_a, sin_a, cos_b, sin_b, *blobs, *extra)
        ctx.mark_non_differentiable(*blobs)

    @staticmethod
    def backward(ctx, dq, dk, dv, *_):
        (
            wq_a,
            wk_a,
            wq_b,
            wk_b,
            cos_a,
            sin_a,
            cos_b,
            sin_b,
            mixed_qkv,
            q_rstd_a,
            k_rstd_a,
            q_rstd_b,
            k_rstd_b,
            xc_a,
            xcs_a,
            wc_a,
            wcs_a,
            xc_b,
            xcs_b,
            wc_b,
            wcs_b,
            *fused,
        ) = ctx.saved_tensors
        m_a, m_b, s_a = ctx.m_a, ctx.m_b, ctx.s_a
        k, n, h, d = ctx.k, ctx.n, ctx.h, ctx.d
        out_dtype = ctx.out_dtype
        want_bias_grad = ctx.needs_input_grad[4]

        grads = tuple(g.contiguous() for g in (dq, dk, dv))
        outs = []
        for i, (rows, off, wq, wk, cos, sin, q_rstd, k_rstd, xc, xcs, wc, wcs) in enumerate(
            (
                (m_a, slice(0, s_a), wq_a, wk_a, cos_a, sin_a, q_rstd_a, k_rstd_a, xc_a, xcs_a, wc_a, wcs_a),
                (
                    m_b,
                    slice(s_a, None),
                    wq_b,
                    wk_b,
                    cos_b,
                    sin_b,
                    q_rstd_b,
                    k_rstd_b,
                    xc_b,
                    xcs_b,
                    wc_b,
                    wcs_b,
                ),
            )
        ):
            g = tuple(x[off].contiguous().reshape(rows, h * d) for x in grads)
            mq = mixed_qkv[0:m_a] if i == 0 else mixed_qkv[m_a:]
            (g_row, g_row_s, g_col, g_col_s, b_partial, dwq_partial, dwk_partial) = (
                _quantize_mxfp6_qk_norm_rope_bwd(mq, *g, cos, sin, wq, wk, q_rstd, k_rstd, want_bias_grad)
            )
            grad_x = gemm_fp6_impl(
                g_row,
                g_row_s,
                wc,
                wcs,
                rows,
                k,
                n,
                out_dtype,
                _GRAN_VALUE,
                None,
                ctx.weight_is_fp4,
            ).reshape(ctx.shape_a if i == 0 else ctx.shape_b)
            if ctx.fuse_wgrad_accum:
                grad_w = _wgrad_into_main_grad(
                    fused[i], g_col, g_col_s, xc, xcs, n, k, rows, gates().wgrad_a6w4
                )
            else:
                grad_w = gemm_fp6_impl(
                    g_col, g_col_s, xc, xcs, n, k, rows, out_dtype, _GRAN_VALUE, None, gates().wgrad_a6w4
                )
            grad_b = b_partial.sum(0).to(out_dtype) if want_bias_grad else None
            grad_wq = _reduce_grad_into_main_grad(
                wq, dwq_partial, wq.dtype, ctx.fuse_wgrad_accum, dims=(0, 1)
            )
            grad_wk = _reduce_grad_into_main_grad(
                wk, dwk_partial, wk.dtype, ctx.fuse_wgrad_accum, dims=(0, 1)
            )
            outs.append((grad_x, grad_w, grad_b, grad_wq, grad_wk))

        (gx_a, gw_a, gb_a, gwq_a, gwk_a), (gx_b, gw_b, gb_b, gwq_b, gwk_b) = outs
        # Trailing Nones cover cos_a, sin_a, cos_b, sin_b, eps, interleaved,
        # fuse_wgrad_accum, grad_enabled and weight_is_fp4.
        return (
            gx_a,
            gx_b,
            gw_a,
            gw_b,
            gb_a,
            gb_b,
            gwq_a,
            gwk_a,
            gwq_b,
            gwk_b,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class MXFP6MLPProjFunction(torch.autograd.Function):
    """A Flux single block's MLP and attention out-projection, sharing one gradient pack.

    The single block computes ``hidden = mlp(norm) + mlp_bias + proj(attn)``, so autograd
    hands MXFP6MLPFunction.backward (fc2) and out-proj's MXFP6LinearFunction.backward the
    *same* gradient. As two Functions each dual-packed it -- two identical [M, 3072] packs
    per block -- and Inductor materialised ``gate * dy`` once per consumer. AOTAutograd does
    not CSE the backward graph, and dynamo cannot trace a cross-Function cache keyed on the
    tensor, so the sharing has to live inside one Function.

    Forward runs the two existing forwards verbatim and returns both outputs *separately*,
    so Inductor keeps fusing ``mlp + bias + proj`` into the downstream scale_add exactly as
    before. Backward packs the gradient once and runs the two existing backwards on it.

    Precondition, enforced by the caller: the two outputs are only ever summed, so their
    gradients are equal. The backward uses ``grad_mlp`` for both. out-proj must be unbiased
    (the shared pack has no column sums) and take the pure-MXFP6 backward.
    """

    @staticmethod
    def forward(x, w1, b1, w2, o, wp, b2, fuse_wgrad_accum, grad_enabled, weight_is_fp4):
        # b2 (fc2's bias) is not applied here -- fc2 is skip_bias_add, the caller still adds
        # it, detached. It is an input only so the backward can return its gradient from the
        # shared pack's column sums instead of autograd reducing gate*dy a second time.
        mlp = MXFP6MLPFunction.forward(x, w1, b1, w2, fuse_wgrad_accum, grad_enabled, weight_is_fp4)
        proj = MXFP6LinearFunction.forward(
            o, wp, None, False, None, 0, 0, fuse_wgrad_accum, grad_enabled, weight_is_fp4
        )
        # (mlp_out, proj_out, 9 MLP extras, 4 out-proj column blobs)
        return (mlp[0], proj[0]) + tuple(mlp[1:]) + tuple(proj[1:])

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, w1, b1, w2, o, wp, b2, fuse_wgrad_accum, grad_enabled, weight_is_fp4 = inputs
        if not grad_enabled:
            return
        ctx.has_b2 = b2 is not None
        ctx.fuse_wgrad_accum = fuse_wgrad_accum
        ctx.weight_is_fp4 = weight_is_fp4
        ctx.out_dtype = x.dtype
        ctx.mlp_orig_shape = x.shape
        ctx.m = x.numel() // x.shape[-1]
        ctx.k = x.shape[-1]
        ctx.f = w1.shape[0]
        ctx.h = w2.shape[0]
        ctx.proj_orig_shape = o.shape
        ctx.pm = o.numel() // o.shape[-1]
        ctx.pk = o.shape[-1]
        ctx.pn = wp.shape[0]

        y1 = output[2]
        mlp_blobs = tuple(output[3:11])
        proj_blobs = tuple(output[11:15])
        mlp_extra = (w1, w2) if fuse_wgrad_accum else ()
        proj_extra = (wp,) if fuse_wgrad_accum else ()
        # Same saves as the two Functions, concatenated; backward splits them back.
        ctx.n_mlp_saved = 2 + len(mlp_blobs) + len(mlp_extra)
        ctx.save_for_backward(
            y1, b1, *mlp_blobs, *mlp_extra, *proj_blobs, *proj_extra, *((b2,) if b2 is not None else ())
        )
        ctx.mark_non_differentiable(*mlp_blobs, *proj_blobs)

    @staticmethod
    def backward(ctx, grad_mlp, grad_proj, *_):
        saved = ctx.saved_tensors
        if ctx.has_b2:
            b2, saved = saved[-1], saved[:-1]
        mlp_saved, proj_saved = saved[: ctx.n_mlp_saved], saved[ctx.n_mlp_saved :]

        # One pack of the shared gradient.
        if not grad_mlp.is_contiguous():
            grad_mlp = grad_mlp.contiguous()
        g2 = grad_mlp.reshape(-1, ctx.h)
        grad_b2 = None
        if ctx.has_b2:
            # Identity prologue with column sums: fc2's bias gradient is exactly the column
            # sum of this tensor, which the packer is already streaming.
            *g_packed, b2_partial = _quantize_mxfp6_fused_dual(g2, None, None, MXFP6_PROLOGUE_IDENTITY, True)
            g_packed = tuple(g_packed)
            grad_b2 = _reduce_grad_into_main_grad(b2, b2_partial, ctx.out_dtype, ctx.fuse_wgrad_accum)
        else:
            # As MXFP6MLPFunction makes it, so the no-bias path stays bit-identical to it.
            g_packed = _quantize_mxfp6_dual(g2)

        grad_x, grad_w1, grad_b1, grad_w2 = _mlp_backward(
            mlp_saved,
            grad_mlp,
            ctx.m,
            ctx.k,
            ctx.f,
            ctx.h,
            ctx.out_dtype,
            ctx.mlp_orig_shape,
            ctx.fuse_wgrad_accum,
            ctx.weight_is_fp4,
            ctx.needs_input_grad[2],
            g2_packed=g_packed,
        )
        grad_o, grad_wp, _ = _linear_mxfp6_backward(
            proj_saved,
            g2,
            ctx.pm,
            ctx.pn,
            ctx.pk,
            ctx.out_dtype,
            ctx.proj_orig_shape,
            ctx.fuse_wgrad_accum,
            ctx.weight_is_fp4,
            False,
            g_packed=g_packed,
        )
        # Trailing Nones cover fuse_wgrad_accum, grad_enabled and weight_is_fp4.
        return grad_x, grad_w1, grad_b1, grad_w2, grad_o, grad_wp, grad_b2, None, None, None


def mlp_proj_shared(mlp, proj, x, o):
    """Run a single block's MLP and out-projection as one MXFP6MLPProjFunction.

    Returns ``(mlp_out, mlp_bias, proj_out)`` shaped like ``mlp(x)`` and ``proj(o)``, or
    None when the pair is not eligible -- the caller then runs them separately, as before.
    The caller guarantees ``mlp_out`` and ``proj_out`` are only summed.
    """
    if not gates().shared_grad_pack:
        return None
    if not getattr(mlp, "_fused_epilogue", False):
        return None
    if not isinstance(proj, MXFP6RowParallelLinear) or proj.bias is not None:
        return None
    if getattr(proj, "_backward_is_fp8", False) or proj._fuse_wgrad_accum != mlp.linear_fc1._fuse_wgrad_accum:
        return None
    if proj._weight_is_fp4 != _resolve_weight_is_fp4(mlp.config):
        return None
    fuse_wgrad_accum = mlp.linear_fc1._fuse_wgrad_accum
    b2 = mlp.linear_fc2.bias if gates().shared_grad_pack_bias else None
    if fuse_wgrad_accum:
        # The same claims the two paths make separately (MXFP6FusedMLP.forward and
        # _mxfp6_forward_impl): the backward writes these straight into main_grad.
        claimed = [mlp.linear_fc1.weight, mlp.linear_fc2.weight, proj.weight]
        if gates().fused_small_grads:
            claimed += [b for b in (mlp.linear_fc1.bias, b2) if b is not None]
        _claim_main_grad(*claimed)
    out = MXFP6MLPProjFunction.apply(
        x,
        mlp.linear_fc1.weight,
        mlp.linear_fc1.bias,
        mlp.linear_fc2.weight,
        o,
        proj.weight,
        b2,
        fuse_wgrad_accum,
        torch.is_grad_enabled(),
        proj._weight_is_fp4,
    )
    # With b2 owned by the Function, the caller's bias add must not open a second gradient
    # path into it -- that path is exactly the reduction this removes.
    return out[0], (b2.detach() if b2 is not None else mlp.linear_fc2.bias), out[1]


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
    # Validated in BaseDiffusionConfig.__post_init__; this is just the read.
    return gates().fused_mlp


class MXFP6FusedMLP(MLP):
    """MLP whose bias-add + GELU is folded into the MXFP6 packer, in both directions.

    Drop-in for ``MLP``: same submodules, same ``(output, output_bias)`` return, same
    parameters and state dict. Only ``forward`` differs, and it defers to ``MLP.forward``
    for anything the fused path does not cover, so a configuration it cannot handle is
    slow rather than wrong.

    ``mxfp6_fused_mlp: off`` forces the stock path for A/B comparison; ``on`` makes
    an unusable configuration an error instead of a silent fallback.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mode = _fused_mlp_mode()
        reason = "disabled by config" if mode == "off" else _fused_mlp_unusable_reason(self)
        self._fused_epilogue = reason == ""

        if not self._fused_epilogue and mode == "on":
            raise RuntimeError(f"mxfp6_fused_mlp=on but the fused MLP is unusable: {reason}")
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
            # fc1's bias is claimed too: its backward now adds straight into main_grad, so
            # the DDP hook must skip its own add_ or the placeholder handed to autograd
            # would be summed on top of a gradient that is already there.
            claimed = [self.linear_fc1.weight, self.linear_fc2.weight]
            if gates().fused_small_grads and self.linear_fc1.bias is not None:
                claimed.append(self.linear_fc1.bias)
            _claim_main_grad(*claimed)

        output = MXFP6MLPFunction.apply(
            hidden_states,
            self.linear_fc1.weight,
            self.linear_fc1.bias,
            self.linear_fc2.weight,
            fuse_wgrad_accum,
            torch.is_grad_enabled(),
            _resolve_weight_is_fp4(self.config),
        )[0]

        # fc2 is built with skip_bias_add=True, so MLP's contract is to hand its bias back
        # unadded for the caller to fuse into a residual.
        return output, self.linear_fc2.bias


def grouped_mlp_pair(mlp_a, mlp_b, x_a, x_b):
    """Run a joint block's two MLPs as one grouped pair.

    Returns ``((out_a, bias_a), (out_b, bias_b))`` shaped exactly like two ``mlp(x)`` calls,
    or ``None`` when this pair is not eligible -- in which case the caller runs the two
    MLPs separately, as before. Every rejection here is a property of the pair rather than
    of the configuration, so the fallback is per block and cannot half-apply.
    """
    if not gates().grouped_mlp or _grouped_mlp_unavailable_reason() is not None:
        return None
    if mlp_a is None or mlp_b is None:
        return None
    # Both sides must be taking the fused-epilogue path, or their outputs would not be
    # comparable operand for operand.
    if not getattr(mlp_a, "_fused_epilogue", False) or not getattr(mlp_b, "_fused_epilogue", False):
        return None
    # The grouped kernel splits its M range in half, so the two streams must contribute
    # equal row counts, and the weights must agree in shape for one N/K to describe both.
    if x_a.shape != x_b.shape:
        return None
    if (
        mlp_a.linear_fc1.weight.shape != mlp_b.linear_fc1.weight.shape
        or mlp_a.linear_fc2.weight.shape != mlp_b.linear_fc2.weight.shape
    ):
        return None
    fuse_a = mlp_a.linear_fc1._fuse_wgrad_accum
    if fuse_a != mlp_b.linear_fc1._fuse_wgrad_accum:
        return None
    if fuse_a:
        claimed = [
            mlp_a.linear_fc1.weight,
            mlp_a.linear_fc2.weight,
            mlp_b.linear_fc1.weight,
            mlp_b.linear_fc2.weight,
        ]
        # Same reason as the ungrouped wrapper: fc1's bias gradient now lands in main_grad
        # directly, so the hook must not add the placeholder on top of it.
        if gates().fused_small_grads:
            for _m in (mlp_a, mlp_b):
                if _m.linear_fc1.bias is not None:
                    claimed.append(_m.linear_fc1.bias)
        _claim_main_grad(*claimed)
    out_a, out_b = MXFP6GroupedMLPFunction.apply(
        x_a,
        x_b,
        mlp_a.linear_fc1.weight,
        mlp_b.linear_fc1.weight,
        mlp_a.linear_fc1.bias,
        mlp_b.linear_fc1.bias,
        mlp_a.linear_fc2.weight,
        mlp_b.linear_fc2.weight,
        fuse_a,
        torch.is_grad_enabled(),
        _resolve_weight_is_fp4(mlp_a.config),
    )[:2]
    # fc2 is built with skip_bias_add=True, so each MLP's contract is to hand its bias back
    # unadded for the caller to fuse into the residual.
    return (out_a, mlp_a.linear_fc2.bias), (out_b, mlp_b.linear_fc2.bias)


def grouped_mlp_enabled() -> bool:
    """Is the grouped joint-block MLP switched on and supported by this build?

    Separate from ``grouped_mlp_pair`` because the call site has to decide whether to hoist
    the context stream's layernorm *before* it has the tensors to test eligibility on, and
    with the gate off that hoist must not happen at all.
    """
    return gates().grouped_mlp and _grouped_mlp_unavailable_reason() is None
