# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Use Turbo tensorwise FP8 for the training vocabulary projection at TP=1."""

from functools import wraps

import torch

from primus.core.utils.module_utils import log_rank_0, log_rank_last

_NATIVE_HEAD_ARGUMENTS = {
    "bias",
    "gradient_accumulation_fusion",
    "allreduce_dgrad",
    "sequence_parallel",
    "grad_output_buffer",
    "wgrad_deferral_limit",
    "tp_group",
    "gtp_remat_size",
}


def _fp8_projection(input, weight, config):
    from primus_turbo.pytorch.ops import gemm_fp8

    logits = gemm_fp8(
        input.reshape(-1, input.shape[-1]),
        weight,
        trans_b=True,
        out_dtype=torch.bfloat16,
        config=config,
    )
    return logits.view(*input.shape[:-1], weight.shape[0])


def _install_on_head(head, config):
    original = head._forward_impl
    logged = False

    @wraps(original)
    def forward_impl(input, weight, *args, **kwargs):
        nonlocal logged
        # Keep the original parameter (including a shared embedding) as the autograd
        # leaf. Megatron DDP accumulates its ordinary wgrad into the main buffer.
        if (
            not args
            and kwargs.keys() == _NATIVE_HEAD_ARGUMENTS
            and head.training
            and torch.is_grad_enabled()
            and weight.requires_grad
            and input.is_cuda
            and input.dtype == weight.dtype == torch.bfloat16
            and kwargs["bias"] is None
            and not kwargs["gradient_accumulation_fusion"]
            and not kwargs["allreduce_dgrad"]
            and not kwargs["sequence_parallel"]
            and kwargs["grad_output_buffer"] is None
            and kwargs["wgrad_deferral_limit"] is None
            and kwargs["gtp_remat_size"] == 1
        ):
            output = _fp8_projection(input, weight, config)
            if not logged:
                log_rank_last(f"[FP8 LM head active] input={tuple(input.shape)} weight={tuple(weight.shape)}")
                logged = True
            return output
        return original(input, weight, *args, **kwargs)

    head._forward_impl = forward_impl


def configure_fp8_lm_head(args) -> None:
    """Enable the requested projection during Megatron argument validation."""
    recipe = getattr(args.fp8_recipe, "value", args.fp8_recipe)
    if not (
        args.enable_primus_turbo
        and args.bf16
        and args.fp8 == "hybrid"
        and recipe == "tensorwise"
        and args.tensor_model_parallel_size == 1
        and not args.gradient_accumulation_fusion
        and not args.defer_embedding_wgrad_compute
    ):
        raise ValueError(
            "turbo_fp8_lm_head requires Turbo, BF16, hybrid tensorwise FP8, TP1, "
            "gradient_accumulation_fusion=false and defer_embedding_wgrad_compute=false."
        )
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.training import get_args
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig,
        Format,
        check_fp8_support,
    )
    from primus_turbo.pytorch.ops import gemm_fp8

    if not callable(gemm_fp8):
        raise RuntimeError("turbo_fp8_lm_head requires Turbo's gemm_fp8 operation.")
    supported, reason = check_fp8_support()
    if not supported:
        raise ValueError(reason)
    original = GPTModel.__init__
    if getattr(original, "_primus_fp8_lm_head", False):
        return
    config = Float8QuantConfig(format=Format.HYBRID)

    @wraps(original)
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.post_process and getattr(get_args(), "turbo_fp8_lm_head", False):
            if type(self.output_layer) is not ColumnParallelLinear:
                raise ValueError("turbo_fp8_lm_head requires a native Megatron ColumnParallelLinear.")
            _install_on_head(self.output_layer, config)

    init._primus_fp8_lm_head = True
    GPTModel.__init__ = init
    log_rank_0(
        "[Patch:megatron.turbo.fp8_lm_head] Enabled tensorwise hybrid FP8 training vocabulary projection"
    )
