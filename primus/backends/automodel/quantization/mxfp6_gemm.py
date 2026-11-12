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

import torch

from primus.backends.automodel.quantization import _fp6_common


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
        from primus_turbo.pytorch.ops.gemm_fp6 import gemm_fp6

        out = gemm_fp6(
            x,
            weight,
            trans_a=False,
            trans_b=True,
            out_dtype=x.dtype,
            config=fp6_config,
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
