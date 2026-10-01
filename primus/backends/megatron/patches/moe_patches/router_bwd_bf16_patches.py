###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mixed-precision MoE router backward with direct FP32 wgrad accumulation.

Megatron keeps FP32 router logits while the router input and parameter are
typically BF16.  Its stock backward promotes both large operands to FP32.  The
opt-in Primus path keeps the forward in FP32, but runs the backward GEMMs with
BF16 operands:

* dX is emitted in BF16.
* dW uses FP32 accumulation/output.
* With gradient-accumulation fusion, dW is written directly into the
  parameter's FP32 ``main_grad`` buffer.

The final point avoids materializing a temporary FP32 dW, narrowing it to a
BF16 ``param.grad``, and then widening/adding it into FP32 ``main_grad``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enabled(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "moe_router_bwd_bf16", False))


def _uses_bf16_router_backward(ctx, te_general_gemm: Callable | None) -> bool:
    """Return whether this invocation matches the optimized precision contract."""
    return (
        te_general_gemm is not None
        and ctx.router_dtype == torch.float32
        and ctx.input_dtype == torch.bfloat16
        and ctx.weight_dtype == torch.bfloat16
    )


def _resolve_fp32_main_grad(weight: torch.Tensor) -> torch.Tensor | None:
    """Return a writable contiguous FP32 main-grad target, when one exists."""
    if hasattr(weight, "__fsdp_param__"):
        main_grad = weight.get_main_grad()
        weight.main_grad = main_grad
    else:
        main_grad = getattr(weight, "main_grad", None)

    if not isinstance(main_grad, torch.Tensor):
        return None
    if main_grad.dtype != torch.float32 or not main_grad.is_contiguous():
        return None
    return main_grad


def _te_general_gemm_accumulate(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    layout: str,
    out: torch.Tensor,
    grad: bool,
    accumulate: bool,
):
    """Call TE's underlying GEMM with an explicit output/accumulate contract.

    The Megatron revision pinned by Primus already exposes ``out`` but hardcodes
    ``accumulate=False`` in its wrapper.  ROCm Transformer Engine supports both
    arguments, so keep this compatibility shim local to the Primus patch.
    """
    from megatron.core.extensions import transformer_engine as te_ext

    kwargs = {
        "out_dtype": out_dtype,
        "quantization_params": None,
        "gelu": None,
        "gelu_in": None,
        "accumulate": accumulate,
        "layout": layout,
        "out": out,
        "bias": None,
        "use_split_accumulator": False,
        "grad": grad,
        "ub": None,
        "ub_type": None,
        "extra_output": None,
        "bulk_overlap": False,
    }

    # TE revisions disagree on whether workspace is an explicit parameter.
    # Match Primus' existing general-GEMM workspace compatibility behavior.
    try:
        accepts_workspace = (
            "workspace" in inspect.signature(te_ext.general_gemm).parameters
        )
    except (TypeError, ValueError):
        accepts_workspace = False
    workspace_helper = getattr(te_ext, "_get_workspace", None)
    if accepts_workspace and workspace_helper is not None:
        kwargs["workspace"] = workspace_helper()

    return te_ext.general_gemm(a, b, **kwargs)


def _get_dummy_wgrad(
    shape: list[int], dtype: torch.dtype, zero: bool = False
) -> torch.Tensor:
    """Return TE's cached dummy gradient, with a compatibility fallback."""
    try:
        from transformer_engine.pytorch.module.base import get_dummy_wgrad

        return get_dummy_wgrad(shape, dtype, zero=zero)
    except ImportError:
        dummy = torch.empty(shape, dtype=dtype, device="cuda", requires_grad=False)
        if zero:
            dummy.zero_()
        return dummy


def _router_backward_bf16(
    ctx,
    grad_output: torch.Tensor,
    *,
    te_general_gemm: Callable,
    te_general_gemm_accumulate: Callable = _te_general_gemm_accumulate,
    get_dummy_wgrad: Callable = _get_dummy_wgrad,
    fuse_main_grad: bool,
):
    """Execute the BF16 router backward selected by the runtime patch."""
    inp, weight, bias = ctx.saved_tensors
    inp_shape = inp.shape
    grad_shape = grad_output.shape
    inp = inp.view(-1, inp_shape[-1])
    grad_output = grad_output.view(-1, grad_shape[-1])
    grad_output_bf16 = grad_output.to(torch.bfloat16)

    grad_input = te_general_gemm(
        weight,
        grad_output_bf16,
        ctx.input_dtype,
        layout="NN",
        grad=True,
    )[0].to(ctx.input_dtype)

    main_grad = _resolve_fp32_main_grad(weight) if fuse_main_grad else None
    if main_grad is not None:
        te_general_gemm_accumulate(
            inp,
            grad_output_bf16,
            out_dtype=main_grad.dtype,
            layout="NT",
            out=main_grad,
            grad=True,
            accumulate=not getattr(weight, "overwrite_main_grad", False),
        )
        if hasattr(weight, "overwrite_main_grad"):
            weight.overwrite_main_grad = False

        if hasattr(weight, "grad_added_to_main_grad"):
            grad_weight = get_dummy_wgrad(
                list(main_grad.shape),
                ctx.weight_dtype,
                zero=getattr(weight, "zero_out_wgrad", False),
            )
            weight.grad_added_to_main_grad = True
        else:
            grad_weight = None
    else:
        # Keep FP32 accumulation/output for the token reduction even when no
        # main-grad target is available.  The autograd-facing gradient must
        # match the BF16 parameter dtype.
        grad_weight = te_general_gemm(
            inp,
            grad_output_bf16,
            ctx.router_dtype,
            layout="NT",
            grad=True,
        )[0].to(ctx.weight_dtype)

    grad_bias = (
        grad_output.sum(dim=0).to(ctx.weight_dtype) if bias is not None else None
    )
    grad_input = grad_input.view(*inp_shape)
    return grad_input, grad_weight, grad_bias, None


@register_patch(
    "megatron.moe.router_bwd_bf16",
    backend="megatron",
    phase="before_train",
    description="Run the FP32 MoE router backward with BF16 operands and direct FP32 main-grad accumulation",
    condition=_enabled,
)
def patch_router_bwd_bf16(ctx: PatchContext) -> None:
    """Install the opt-in mixed-precision router backward implementation."""
    from megatron.core.transformer.moe import moe_utils

    router_function = moe_utils.RouterGatingLinearFunction
    if not hasattr(router_function, "_primus_original_backward"):
        router_function._primus_original_backward = router_function.backward

    original_backward = router_function._primus_original_backward
    router_function._primus_fuse_router_main_grad = bool(
        getattr(get_args(ctx), "gradient_accumulation_fusion", False)
    )

    def backward(autograd_ctx, grad_output):
        te_general_gemm = moe_utils.te_general_gemm
        if not _uses_bf16_router_backward(autograd_ctx, te_general_gemm):
            return original_backward(autograd_ctx, grad_output)
        return _router_backward_bf16(
            autograd_ctx,
            grad_output,
            te_general_gemm=te_general_gemm,
            fuse_main_grad=router_function._primus_fuse_router_main_grad,
        )

    router_function.backward = staticmethod(backward)
    mode = (
        "direct FP32 main_grad"
        if router_function._primus_fuse_router_main_grad
        else "BF16 param.grad"
    )
    log_rank_0(
        f"[Patch:megatron.moe.router_bwd_bf16] Installed router backward ({mode})"
    )
