###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron pipeline parallel (PP) warmup patches.

This module patches ``megatron.training.training.train`` so that when
``args.pp_warmup`` is True (config: ``pp_warmup`` in primus_megatron_module.yaml),
PP warmup runs once immediately before the first call to ``train()``.
"""

import inspect

import torch

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


def run_pp_warmup(model, config, args, optimizer, timers):
    """
    Run pipeline parallel warmup: one forward/backward over decoder layers per
    model chunk with no_sync, then zero_grad and empty_cache.
    """
    timers("pp-warmup", log_level=0).start(barrier=True)
    for model_chunk in model:
        with model_chunk.no_sync():
            if model_chunk.use_forward_hook:
                model_chunk.disable_forward_pre_hook()
            dtype = torch.float32
            if config.bf16:
                dtype = torch.bfloat16
            elif config.fp16:
                dtype = torch.float16
            seq_len = args.seq_length // args.tensor_model_parallel_size // args.context_parallel_size
            for layer in model_chunk.module.module.decoder.layers:
                dummy_input = torch.randn(seq_len, 1, config.hidden_size, device="cuda", dtype=dtype)
                attention_mask = (
                    torch.tril(torch.ones((seq_len, seq_len), device="cuda")).unsqueeze(0).unsqueeze(0) == 0
                )
                dummy_output, _ = layer.forward(hidden_states=dummy_input, attention_mask=attention_mask)
                dummy_output.backward(torch.ones_like(dummy_output))
            if model_chunk.use_forward_hook:
                model_chunk.enable_forward_pre_hook()
            optimizer.zero_grad()
    torch.cuda.empty_cache()
    timers("pp-warmup").stop()
    timers.log(["pp-warmup"], barrier=True)


@register_patch(
    "megatron.training.pp_warmup.wrap_train",
    backend="megatron",
    phase="before_train",
    description="Wrap train() to run PP warmup before the first train() call when args.pp_warmup is True.",
)
def patch_train_with_pp_warmup(ctx: PatchContext) -> None:
    """
    Replace ``megatron.training.training.train`` with a wrapper that runs
    PP warmup once (when args.pp_warmup is True) before calling the original train.
    """
    import megatron.training.training as training  # type: ignore
    from megatron.training.global_vars import get_args as get_megatron_args
    from megatron.training.global_vars import get_timers

    original_train = training.train

    if getattr(original_train, "_primus_pp_warmup_wrapped", False):
        return

    train_signature = inspect.signature(original_train)

    def _train_with_pp_warmup(*call_args, **call_kwargs):
        megatron_args = get_megatron_args()
        if getattr(megatron_args, "pp_warmup", True):
            bound = None
            try:
                bound = train_signature.bind_partial(*call_args, **call_kwargs)
            except TypeError:
                # Fall through and call the original train; signature mismatch can
                # happen across Megatron versions.
                pass

            model = bound.arguments.get("model") if bound else None
            config = bound.arguments.get("config") if bound else None
            optimizer = bound.arguments.get("optimizer") if bound else None

            if model is not None and config is not None and optimizer is not None:
                run_pp_warmup(model, config, megatron_args, optimizer, get_timers())
            else:
                log_rank_0(
                    "[Patch:megatron.pp_warmup] Skip warmup for this train() call: "
                    "failed to resolve model/config/optimizer from current train signature."
                )

        return original_train(*call_args, **call_kwargs)

    setattr(_train_with_pp_warmup, "_primus_pp_warmup_wrapped", True)
    training.train = _train_with_pp_warmup
    log_rank_0(
        f"[Patch:megatron.pp_warmup] Wrapped train(); PP warmup runs before train when args.pp_warmup = True"
    )
