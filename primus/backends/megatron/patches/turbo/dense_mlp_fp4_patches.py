###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Dense MXFP4 SwiGLU MLP patch
============================

Replaces Megatron ``MLP.forward``'s fused-LN ``linear_fc1`` + SwiGLU +
``linear_fc2`` with Primus-Turbo ``mlp_fp4`` (dense ``kernel_gemm_4w``
+ ``StoreCSwiGLU`` for MLP-up; ``gemm_fp4`` for fc2 / backward). TE's spec
bakes pre-MLP RMSNorm into ``linear_fc1`` as ``layer_norm_weight``; that
norm still runs here so DDP overlap hooks see it. QKV and O-proj stay on
``PrimusTurboLinear``. Opt-in:

    YAML  turbo_fused_gemm: true
    env   PRIMUS_TURBO_FUSED_DENSE_MLP=1   (overrides YAML)

Requires ``fp4`` + ``use_turbo_gemm``, SwiGLU, no linear bias, and no
per-token MoE scale.
"""

from __future__ import annotations

import os

import torch

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.mlp.turbo_dense_mlp_fp4"


def _env_flag(name: str) -> bool | None:
    if name not in os.environ:
        return None
    return os.environ[name] not in ("0", "", "false", "False")


def _fused_dense_mlp_requested(args) -> bool:
    env = _env_flag("PRIMUS_TURBO_FUSED_DENSE_MLP")
    if env is None:
        env = _env_flag("PRIMUS_TURBO_FUSED_GEMM")
    if env is not None:
        return env
    return bool(getattr(args, "turbo_fused_gemm", False) or getattr(args, "use_turbo_fused_dense_mlp", False))


def _should_enable(mlp, args) -> bool:
    if not _fused_dense_mlp_requested(args):
        return False
    if not bool(getattr(args, "fp4", False)):
        return False
    if not bool(getattr(args, "use_turbo_gemm", False)):
        return False
    cfg = mlp.config
    if not getattr(cfg, "gated_linear_unit", False):
        return False
    if getattr(cfg, "add_bias_linear", False):
        return False
    return True


def _install_ddp_grad_ready_debug() -> None:
    if os.environ.get("PRIMUS_DDP_GRAD_READY_DEBUG", "0") not in ("1", "true", "True"):
        return
    from collections import Counter

    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

    if getattr(_ParamAndGradBucketGroup.reset, "_primus_ddp_debug", False):
        return

    orig_reset = _ParamAndGradBucketGroup.reset

    def _reset(self):
        if self.is_first_batch and len(self.per_param_grad_ready_counts) > 0:
            if len(self.per_param_grad_ready_counts) != len(self.params):
                missing = [p for p in self.params if p not in self.per_param_grad_ready_counts]
                shapes = Counter(tuple(p.shape) for p in missing)
                log_rank_0(
                    f"[DDP-DEBUG] grad-ready {len(self.per_param_grad_ready_counts)}/"
                    f"{len(self.params)} missing={len(missing)} shapes={dict(shapes)}"
                )
        return orig_reset(self)

    _reset._primus_ddp_debug = True
    _ParamAndGradBucketGroup.reset = _reset
    log_rank_0("[Patch:megatron.mlp.turbo_dense_mlp_fp4] DDP grad-ready debug enabled.")


def _apply_fc1_fused_norm(fc1, hidden_states, config):
    """Run the RMS/LayerNorm TE fused into ``linear_fc1``, if present.

    Megatron's TE spec uses ``LayerNormLinear`` for MLP fc1 and Identity for
    ``pre_mlp_layernorm``. Skipping ``linear_fc1.forward`` therefore also
    skipped the pre-MLP norm, so ``layer_norm_weight`` never got a gradient
    and ``overlap_grad_reduce`` failed the first-batch golden-count assert.
    """
    if getattr(fc1, "_skip_fused_norm", False):
        return hidden_states
    ln_w = getattr(fc1, "layer_norm_weight", None)
    if ln_w is None:
        return hidden_states
    eps = getattr(fc1, "eps", config.layernorm_epsilon)
    if config.normalization == "RMSNorm":
        from primus_turbo.pytorch.ops.normalization import rmsnorm

        return rmsnorm(hidden_states, ln_w, eps)
    if config.normalization == "LayerNorm":
        import torch.nn.functional as F

        return F.layer_norm(
            hidden_states,
            [hidden_states.size(-1)],
            ln_w,
            getattr(fc1, "layer_norm_bias", None),
            eps,
        )
    raise RuntimeError(f"turbo_fused_gemm: unsupported normalization {config.normalization!r}")


def _run_module_forward_pre_hooks(module, args):
    """Run ``nn.Module`` forward-pre-hooks without calling ``forward``.

    DistributedDataParallel's ``overlap_param_gather`` waits on / dispatches
    the parameter all-gather from each submodule's pre-hook. The fused MLP
    never enters ``linear_fc1`` / ``linear_fc2``.forward, so those hooks must
    still run or the next layer's QKV hits ``param_gather_handle is None``.
    """
    if not isinstance(args, tuple):
        args = (args,)
    for hook in module._forward_pre_hooks.values():
        result = hook(module, args)
        if result is not None:
            args = result if isinstance(result, tuple) else (result,)
    kw_hooks = getattr(module, "_forward_pre_hooks_with_kwargs", None)
    if kw_hooks:
        kwargs: dict = {}
        for hook in kw_hooks.values():
            result = hook(module, args, kwargs)
            if result is not None:
                args, kwargs = result
    return args


def _wrap_prequant(x, prequant, config):
    """Wrap a fused-norm ``(row, row_scale, col, col_scale)`` as MXFP4 operands.

    ``row`` is the forward operand; ``col`` is the RHT col-wise wgrad operand.
    """
    from primus_turbo.pytorch.core.low_precision import (
        MXFP4_BLOCK_SIZE,
        ScalingGranularity,
        ScalingRecipe,
        float4_e2m1fn_x2,
    )
    from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor

    row, row_scale, col, col_scale = prequant
    hidden = x.shape[-1]
    rows = x.reshape(-1, hidden).shape[0]

    def _wrap(data, scale, shape, axis, recipe):
        return QuantizedTensor(
            data.contiguous(),
            scale.contiguous(),
            shape=shape,
            orig_dtype=x.dtype,
            dest_dtype=float4_e2m1fn_x2,
            granularity=ScalingGranularity.MX_BLOCKWISE,
            block_size=MXFP4_BLOCK_SIZE,
            scaling_recipe=recipe,
            scale_rounding_mode=config.scale_rounding_mode,
            quantized_axis=axis,
        )

    return (
        _wrap(row, row_scale, torch.Size((rows, hidden)), -1, ScalingRecipe()),
        _wrap(col, col_scale, torch.Size((hidden, rows)), -2, ScalingRecipe(use_rht=True)),
    )


class _PrequantSTE(torch.autograd.Function):
    """Keep the MLP dgrad connected to the BF16 norm output ``x``."""

    @staticmethod
    def forward(ctx, x, prequant, config):
        ctx.input_shape = x.shape
        ctx.set_materialize_grads(False)
        row, col = _wrap_prequant(x, prequant, config)
        # The col-wise operand only feeds the weight gradient.
        ctx.mark_non_differentiable(col)
        return row, col

    @staticmethod
    def backward(ctx, grad_row, grad_col):
        del grad_col
        grad_x = None if grad_row is None else grad_row.reshape(ctx.input_shape)
        return grad_x, None, None


def _forward_turbo_dense_mlp_fp4(mlp, hidden_states):
    from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensorPair
    from primus_turbo.pytorch.ops.mlp_fp4 import mlp_fp4

    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboLowPrecisionGlobalStateManager,
        _fuse_wgrad_accum_pattern,
    )

    # Same objects DDP registered AccumulateGrad hooks on (not TE aliases).
    w1 = mlp.linear_fc1._parameters["weight"]
    w2 = mlp.linear_fc2._parameters["weight"]
    # Keep DistOpt param-gather overlap in registration order: wait fc1, dispatch
    # fc2, wait fc2, dispatch the next layer — then run the fused GEMMs.
    _run_module_forward_pre_hooks(mlp.linear_fc1, (hidden_states,))
    _run_module_forward_pre_hooks(mlp.linear_fc2, (hidden_states,))
    hidden_states = _apply_fc1_fused_norm(mlp.linear_fc1, hidden_states, mlp.config)
    leading = hidden_states.shape[:-1]
    x = hidden_states.reshape(-1, hidden_states.shape[-1])

    quant = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
    assert quant is not None and quant.mxfp4_scaling(), "turbo_fused_gemm requires MXFP4 Turbo autocast"
    config = quant.data()
    pre = getattr(mlp.linear_fc1, "_prequant_x", None)
    if pre is not None:
        x = QuantizedTensorPair(*_PrequantSTE.apply(x, pre, config))
    y = mlp_fp4(
        x,
        w1,
        w2,
        trans_w1=True,
        trans_w2=True,
        out_dtype=hidden_states.dtype,
        config=config,
        fuse_wgrad_accum_pattern=_fuse_wgrad_accum_pattern(mlp.config, w1),
        activation="silu",
    )
    return y.view(*leading, y.shape[-1]), None


def _install_dense_mlp_fp4_patch() -> None:
    from megatron.core.transformer.mlp import MLP
    from megatron.training import get_args as _get_args

    if is_patched(MLP, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] MLP already patched; skipping.")
        return

    _install_ddp_grad_ready_debug()

    orig_init = MLP.__init__
    orig_forward = MLP.forward
    orig_backward_dw = MLP.backward_dw

    def _init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._use_turbo_dense_mlp_fp4 = _should_enable(self, _get_args())

    def _forward(self, hidden_states, per_token_scale=None, **kwargs):
        if getattr(self, "_use_turbo_dense_mlp_fp4", False) and per_token_scale is None:
            return _forward_turbo_dense_mlp_fp4(self, hidden_states)
        return orig_forward(self, hidden_states, per_token_scale, **kwargs)

    def _backward_dw(self):
        if getattr(self, "_use_turbo_dense_mlp_fp4", False):
            return
        return orig_backward_dw(self)

    MLP.__init__ = _init
    MLP.forward = _forward
    MLP.backward_dw = _backward_dw
    mark_patched(MLP, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] MLP.forward routes dense SwiGLU through "
        "primus_turbo.ops.mlp_fp4 when turbo_fused_gemm is set."
    )


def _patch_condition(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return _fused_dense_mlp_requested(args) and bool(getattr(args, "fp4", False))


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Route dense MLP SwiGLU through FlyDSL dense MXFP4 kernel_gemm_4w + "
        "StoreCSwiGLU when turbo_fused_gemm is set."
    ),
    priority=60,
    condition=_patch_condition,
)
def patch_turbo_dense_mlp_fp4(ctx: PatchContext) -> None:
    _install_dense_mlp_fp4_patch()
