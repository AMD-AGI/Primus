###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MXFP6-forward, MXFP4-backward GEMM for the AutoModel linear swap.

Turbo's autograd-aware ``gemm_fp6`` requires M, N and K to be multiples of 256:
the dimensions exchange roles in dgrad and wgrad.  A Linear's feature dimensions
are checked before replacement.  Its token count is known only at runtime, so
this module zero-pads tokens and slices the output back.

The padding composition is exact.  Autograd pads the sliced output gradient
with zeros, and those added rows contribute zero to the weight gradient.
"""
from __future__ import annotations

import os

import torch

from primus.backends.automodel.quantization import _fp6_common

_SEEN_SHAPES: set[tuple[str, int, int, int]] = set()


def _record_shape(family: str, m: int, n: int, k: int) -> None:
    """Print each model GEMM shape once when the launch witness requests it."""
    key = (family, m, n, k)
    if os.environ.get("PRIMUS_MXFP6_SHAPE_LOG") == "1" and key not in _SEEN_SHAPES:
        _SEEN_SHAPES.add(key)
        print(f"PRIMUS_MXFP6_SHAPE family={family} M={m} N={n} K={k}", flush=True)


def _pad_tokens(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return ``x`` padded on its token axis and the original token count."""
    tokens = x.shape[0]
    padded_tokens = _fp6_common.pad_multiple(tokens)
    if padded_tokens == tokens:
        return x, tokens
    return torch.nn.functional.pad(x, (0, 0, 0, padded_tokens - tokens)), tokens


class _MXFP6FwdMXFP4Bwd(torch.autograd.Function):
    """MXFP6 forward with both gradient GEMMs in MXFP4."""

    @staticmethod
    def forward(ctx, x, weight, fp6_config, fp4_config):
        from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import gemm_fp6_impl
        from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import (
            check_mxfp6_support,
            quantize_mxfp6_row,
        )

        supported, reason = check_mxfp6_support(x.device)
        if not supported:
            raise RuntimeError(reason)
        if x.ndim != 2 or weight.ndim != 2:
            raise ValueError(f"MXFP6 linear needs 2D operands, got x={x.ndim}D weight={weight.ndim}D")
        if x.device != weight.device or x.dtype != weight.dtype:
            raise ValueError(
                "MXFP6 linear operands must share device and dtype, got "
                f"x={x.device}/{x.dtype} weight={weight.device}/{weight.dtype}"
            )
        m, k = x.shape
        n, weight_k = weight.shape
        if k != weight_k or any(dim % _fp6_common.MXFP6_TILE_SIZE for dim in (m, n, k)):
            raise ValueError(f"MXFP6 linear needs matching 256-aligned M/N/K, got M={m} N={n} K={k}")

        # Only row packs feed the forward GEMM. Turbo's autograd-aware
        # gemm_fp6 also creates column packs for an MXFP6 backward; this hybrid
        # recipe deliberately uses MXFP4 backward, so those packs are waste.
        x_row, x_scale = quantize_mxfp6_row(x, fp6_config.block_size)
        weight_row, weight_scale = quantize_mxfp6_row(weight, fp6_config.block_size)
        _record_shape("A6W6", m, n, k)
        out = gemm_fp6_impl(
            x_row,
            x_scale,
            weight_row,
            weight_scale,
            m,
            n,
            k,
            x.dtype,
            fp6_config.granularity.value,
        )
        ctx.save_for_backward(x, weight)
        ctx.out_dtype = x.dtype
        ctx.fp4_config = fp4_config
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.core.low_precision import ScalingRecipe
        from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import gemm_fp4_impl
        from primus_turbo.pytorch.ops.quantization import quantize_fp4_with_trans

        x, weight = ctx.saved_tensors
        grad_2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        config = ctx.fp4_config
        preshuffled = config.use_preshuffle
        backend = (BackendType.AITER if preshuffled else BackendType.HIPBLASLT).value
        fp4_dtype = torch.float4_e2m1fn_x2

        # Match Turbo's native MXFP4 backward recipes.  Packing the column
        # directions avoids materialising multi-GB BF16 transposes for every WAN
        # layer while still delegating all quantization and GEMM numerics.
        grad_row_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=config.use_gradient_sr,
            use_rht=False,
            shuffle_scale=preshuffled,
            shuffle_out=False,
        )
        grad_col_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=config.use_gradient_sr,
            use_rht=True,
            shuffle_scale=preshuffled,
            shuffle_out=False,
        )
        g_row, g_row_scale, g_col, g_col_scale = quantize_fp4_with_trans(
            grad_2d,
            fp4_dtype,
            config.granularity,
            block_size=config.block_size,
            scaling_recipe=grad_row_recipe,
            scaling_recipe_for_trans=grad_col_recipe,
        )

        activation_row_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=False,
            use_rht=False,
            shuffle_scale=preshuffled,
            shuffle_out=False,
        )
        activation_col_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=False,
            use_rht=True,
            shuffle_scale=preshuffled,
            shuffle_out=preshuffled,
        )
        _, _, x_col, x_col_scale = quantize_fp4_with_trans(
            x,
            fp4_dtype,
            config.granularity,
            block_size=config.block_size,
            scaling_recipe=activation_row_recipe,
            scaling_recipe_for_trans=activation_col_recipe,
        )

        weight_recipe = ScalingRecipe(
            use_2d_block=True,
            use_sr=False,
            use_rht=False,
            shuffle_scale=preshuffled,
            shuffle_out=preshuffled,
        )
        _, _, weight_col, weight_col_scale = quantize_fp4_with_trans(
            weight,
            fp4_dtype,
            config.granularity,
            block_size=config.block_size,
            scaling_recipe=weight_recipe,
            scaling_recipe_for_trans=weight_recipe,
        )

        common = dict(
            out_dtype=ctx.out_dtype,
            trans_c=False,
            granularity=config.granularity.value,
            default_backend=backend,
            preshuffled=preshuffled,
        )
        # dgrad = grad_out @ weight
        _record_shape("A4W4", grad_2d.shape[0], weight.shape[1], weight.shape[0])
        grad_x = gemm_fp4_impl(
            g_row,
            g_row_scale,
            False,
            weight_col,
            weight_col_scale,
            True,
            **common,
        )
        # wgrad = grad_out.T @ x
        _record_shape("A4W4", weight.shape[0], weight.shape[1], x.shape[0])
        grad_weight = gemm_fp4_impl(
            g_col,
            g_col_scale,
            False,
            x_col,
            x_col_scale,
            True,
            **common,
        )
        return grad_x, grad_weight, None, None


def mxfp6_fwd_mxfp4_bwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    fp6_config,
    fp4_config,
) -> torch.Tensor:
    """Compute ``x @ weight.T`` with MXFP6 forward and MXFP4 backward."""
    padded, real_tokens = _pad_tokens(x)
    out = _MXFP6FwdMXFP4Bwd.apply(padded, weight, fp6_config, fp4_config)
    if padded.shape[0] != real_tokens:
        out = out[:real_tokens]
    return out
