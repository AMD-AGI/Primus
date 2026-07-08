###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from functools import lru_cache
from typing import Any, List

import torch
from megatron.core import parallel_state

from primus.modules.module_utils import log_rank_0


@lru_cache
def produce_attention_sharder(cp_comm_type: str):
    # Import only when needed to avoid import errors if primus_turbo is not available
    try:
        from primus_turbo.pytorch.ops.attention.attention_utils import (
            All2AllAttentionSharder,
        )
    except ImportError:
        raise ImportError("All2AllAttentionSharder not available. Ensure primus_turbo is properly installed.")

    if cp_comm_type == "a2a":
        return All2AllAttentionSharder()
    else:
        raise ValueError(f"Unsupported cp_comm_type: {cp_comm_type}")


def shard_batch_on_this_cp_rank(sharder, batch):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_group = parallel_state.get_context_parallel_group()
    if cp_size > 1:
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 if key != "attention_mask" else 2
                batch[key] = sharder.shard_cp_input([val], cp_group, seq_dim)[0]
    return batch


def apply_torch_compile_if_enabled(model: List[Any], args: Any) -> None:
    """
    Apply torch.compile to model AFTER distributed wrapping and config setup.

    This must be called:
    1. AFTER get_model() completes (which does FSDP/DDP wrapping internally)
    2. AFTER ddp_config is set on the wrapped model
    3. BEFORE optimizer creation

    Args:
        model: List of model modules (already wrapped in DDP/FSDP)
        args: Megatron args object

    Raises:
        Exception: If compilation fails, exception is raised (not caught)
    """
    from megatron.training.utils import unwrap_model

    # Check if compilation is enabled
    if not getattr(args, "enable_torch_compile", False):
        return

    log_rank_0("=" * 80)
    log_rank_0("Applying torch.compile AFTER distributed wrapping...")
    log_rank_0("=" * 80)

    # Import torch_FSDP for type checking
    try:
        from primus.backends.megatron.core.distributed.torch_fully_sharded_data_parallel import (
            PrimusTorchFullyShardedDataParallel as torch_FSDP,
        )
    except ImportError:
        torch_FSDP = None

    for idx, model_module in enumerate(model):
        # Check what type of wrapper we have
        wrapper_type = type(model_module).__name__
        log_rank_0(f"  Model [{idx}] wrapper: {wrapper_type}")

        # Check if wrapped with FSDP and has ddp_config
        if torch_FSDP and isinstance(model_module, torch_FSDP):
            has_config = hasattr(model_module, "ddp_config")
            log_rank_0(f"  FSDP detected - has ddp_config: {has_config}")

            # Check if FSDP wrapper has compile_model method
            if hasattr(model_module, "compile_model"):
                log_rank_0(f"  Calling compile_model on FSDP wrapper...")
                model_module.compile_model()
                log_rank_0(f"    ✓ FSDP wrapper compilation complete")
                continue

        # Unwrap to get the actual model (Flux, GPT, etc.)
        unwrapped_models = unwrap_model([model_module])

        for unwrapped in unwrapped_models:
            model_type = type(unwrapped).__name__

            # Check if the model has a compile_model method
            if hasattr(unwrapped, "compile_model"):
                log_rank_0(f"  Compiling {model_type}...")

                # Apply compilation
                unwrapped.compile_model()

                log_rank_0(f"    ✓ {model_type} compilation complete")
            else:
                log_rank_0(f"  ℹ  {model_type} does not support torch.compile " f"(no compile_model method)")

    log_rank_0("torch.compile application complete")
    log_rank_0("=" * 80)


def apply_torch_compile_to_optimizer_if_enabled(optimizer: Any, args: Any) -> None:
    """
    Apply torch.compile to the optimizer's step() when enable_torch_compile is True.
    Replaces optimizer.step with a compiled version so the trainer needs no changes.
    Uses fullgraph=False for the optimizer step (recommended; allows timers/conditionals).
    """
    if optimizer is None:
        return
    if not getattr(args, "enable_torch_compile", False):
        return
    if not getattr(args, "torch_compile_optimizer", False):
        log_rank_0("  Optimizer compilation disabled (torch_compile_optimizer=False)")
        return

    backend = getattr(args, "torch_compile_backend", "inductor")
    mode = getattr(args, "torch_compile_mode", "default")
    fullgraph = False  # Optimizer step: allow graph breaks (timers, conditionals)

    compile_kwargs = {
        "backend": backend,
        "mode": mode,
        "fullgraph": fullgraph,
    }

    scope = getattr(args, "torch_compile_optimizer_scope", "full")
    log_rank_0(f"Applying torch.compile to optimizer step (scope={scope})...")
    log_rank_0(f"  backend={backend}, mode={mode}, fullgraph={fullgraph}")

    if scope == "inner_only":
        inner_opt = getattr(optimizer, "optimizer", None)
        if inner_opt is None:
            log_rank_0("  ⚠ No inner optimizer found, falling back to full scope")
            scope = "full"
        else:
            inner_opt.step = torch.compile(inner_opt.step, **compile_kwargs)
            log_rank_0("  ✓ Inner optimizer step compiled (clip_grad_norm/DTensor ops excluded)")

    if scope == "full":
        original_step = optimizer.step
        optimizer.step = torch.compile(original_step, **compile_kwargs)
        log_rank_0("  ✓ Optimizer step compiled (full scope)")
