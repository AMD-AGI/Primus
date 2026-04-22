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

import torch

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


def run_pp_warmup(forward_step_func, model, optimizer, config):
    """Run a single fake forward+backward on every pipeline-parallel rank in parallel.

    Why this exists
    ---------------
    The first real training iteration in a pipeline-parallel (PP) run is much
    slower than subsequent iterations because every PP rank lazily initializes a
    large number of things on its first forward/backward pass (CUDA kernels,
    TransformerEngine workspaces, FP8 amax buffers, dropout RNG states, NCCL
    buffers, etc). These initializations happen *serially* across PP ranks since
    downstream ranks block on p2p recv from upstream. On a PP_SIZE=8 run this
    makes iter-1 roughly 8x slower than it needs to be.

    What this function does
    -----------------------
    It performs one forward+backward on every PP rank *in parallel* by bypassing
    p2p communication entirely: each rank fabricates its own input activation
    (on non-first stages) and its own output gradient (on non-last stages) using
    the shapes `get_tensor_shapes` would produce, then runs the model with the
    synthetic tensors. This exercises all lazy-init paths concurrently, so the
    first real iteration no longer pays that cost.

    Correctness guarantees
    ----------------------
    * All RNG state (CPU, CUDA, and Megatron's CudaRNGStatesTracker) is
      snapshotted before the warmup and restored afterwards, so training is
      bit-for-bit identical with/without `--pp-warmup`.
    * Gradient buffers are zeroed both before and after the warmup, and DDP
      reductions are suppressed via `no_sync()`, so no warmup gradient or
      stale all-reduce can leak into the first real step.
    * MoE aux-loss trackers are cleared at the end of warmup.
    * `rerun_state_machine` validation is disabled during warmup so the dummy
      loss does not pollute NaN/Inf/spiky-loss checks.
    * The optimizer step is never called.
    """
    import contextlib
    import time

    from megatron.core import mpu
    from megatron.core.pipeline_parallel.schedules import (
        backward_step as _schedule_backward_step,
    )
    from megatron.core.pipeline_parallel.schedules import deallocate_output_tensor
    from megatron.core.pipeline_parallel.schedules import (
        forward_step as _schedule_forward_step,
    )
    from megatron.core.pipeline_parallel.schedules import get_tensor_shapes
    from megatron.core.pipeline_parallel.utils import (
        is_pp_first_stage,
        is_pp_last_stage,
    )
    from megatron.core.tensor_parallel.random import (
        _get_all_rng_states,
        _set_all_rng_states,
    )
    from megatron.core.transformer.moe.moe_utils import clear_aux_losses_tracker
    from megatron.core.utils import get_pg_size
    from megatron.training.global_vars import get_args

    args = get_args()

    pp_group = mpu.get_pipeline_model_parallel_group()
    pp_size = get_pg_size(pp_group)

    if pp_size == 1:
        log_rank_0("pp_size==1, skipping warmup (nothing to serialize).")
        return

    log_rank_0("entering warmup")
    t0 = time.time()

    # --- 1. Snapshot RNG + a few flags we may mutate -----------------------------
    saved_rng_states = _get_all_rng_states()
    saved_check_nan = args.check_for_nan_in_loss_and_grad
    saved_check_spiky = getattr(args, "check_for_spiky_loss", False)
    saved_curr_iteration = getattr(args, "curr_iteration", None)
    # Suppress NaN/Inf/Spiky validation — the warmup output is meaningless numerically.
    args.check_for_nan_in_loss_and_grad = False
    if hasattr(args, "check_for_spiky_loss"):
        args.check_for_spiky_loss = False
    args.curr_iteration = -1  # signal "not a real iteration" to anything that inspects this

    # --- 2. Zero any existing grad state -----------------------------------------
    for model_chunk in model:
        if hasattr(model_chunk, "zero_grad_buffer"):
            model_chunk.zero_grad_buffer()
    if optimizer is not None:
        optimizer.zero_grad()

    # --- 3. Build a synthetic one-microbatch data iterator -----------------------
    # This matches the dict that `get_batch_on_this_tp_rank` expects on TP rank 0.
    # On intermediate PP stages `get_batch` returns early without touching the
    # iterator, so the same fake batch is fine for every rank.
    device = torch.cuda.current_device()
    mbs = args.micro_batch_size
    seqlen = args.seq_length
    fake_batch = {
        "tokens": torch.zeros(mbs, seqlen, dtype=torch.int64, device=device),
        "labels": torch.zeros(mbs, seqlen, dtype=torch.int64, device=device),
        "loss_mask": torch.ones(mbs, seqlen, dtype=torch.float32, device=device),
        "position_ids": (
            torch.arange(seqlen, dtype=torch.int64, device=device)
            .unsqueeze(0)
            .expand(mbs, seqlen)
            .contiguous()
        ),
    }
    if args.create_attention_mask_in_dataloader:
        fake_batch["attention_mask"] = torch.zeros(mbs, 1, seqlen, seqlen, dtype=torch.bool, device=device)

    # We supply enough repetitions to cover a forward-step call per model chunk.
    # `iter` returns a fresh one-shot iterator; one call per chunk is plenty.
    def _make_fake_iter():
        return iter([fake_batch] * max(1, len(model)))

    # --- 4. Figure out activation shape between PP stages ------------------------
    tp_group = mpu.get_tensor_model_parallel_group()
    cp_group = mpu.get_context_parallel_group()
    tensor_shapes = get_tensor_shapes(
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    activation_dtype = (
        config.pipeline_dtype
        if getattr(config, "pipeline_dtype", None) is not None
        else getattr(config, "params_dtype", torch.float32)
    )

    # --- 5. Run forward+backward on each (virtual) chunk -------------------------
    is_pp_first = is_pp_first_stage(pp_group)
    is_pp_last = is_pp_last_stage(pp_group)
    num_chunks = len(model)
    fake_iter = _make_fake_iter()

    torch.cuda.synchronize()

    for chunk_idx, model_chunk in enumerate(model):
        vp_stage = None if num_chunks == 1 else chunk_idx
        # With VPP, only the very first chunk of pp_rank 0 has pre_process=True,
        # and only the last chunk of the last pp_rank has post_process=True.
        chunk_pre_process = is_pp_first and chunk_idx == 0
        chunk_post_process = is_pp_last and chunk_idx == num_chunks - 1

        # Fabricate the input activation that would normally come from recv_forward.
        if chunk_pre_process:
            input_tensor = None
        else:
            input_tensor = [
                torch.zeros(shape, dtype=activation_dtype, device=device, requires_grad=True)
                for shape in tensor_shapes
            ]

        forward_data_store = []
        output_tensor, _num_tokens = _schedule_forward_step(
            forward_step_func,
            fake_iter,
            model_chunk,
            num_microbatches=1,
            input_tensor=input_tensor,
            forward_data_store=forward_data_store,
            config=config,
            cp_group_size=cp_group.size(),
            collect_non_loss_data=False,
            is_first_microbatch=True,
            current_microbatch=0,
            vp_stage=vp_stage,
            is_last_stage=chunk_post_process,
        )

        # Fabricate the output grad that would normally come from recv_backward.
        if chunk_post_process:
            output_tensor_grad = None
        else:
            ot = output_tensor[0] if isinstance(output_tensor, list) else output_tensor
            output_tensor_grad = [torch.zeros_like(ot)]
            # Mimic the real schedule: deallocate the output so custom_backward is used.
            deallocate_output_tensor(ot, getattr(config, "deallocate_pipeline_outputs", False))

        # Suppress DDP gradient sync — we're only running one microbatch and will
        # throw away the grads immediately afterwards.
        nosync_ctx = model_chunk.no_sync() if hasattr(model_chunk, "no_sync") else contextlib.nullcontext()
        with nosync_ctx:
            _schedule_backward_step(
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                output_tensor_grad=output_tensor_grad,
                model_type=getattr(config, "model_type", None),
                config=config,
            )

        # Free fabricated tensors eagerly before moving on.
        del output_tensor, output_tensor_grad, input_tensor, forward_data_store

    torch.cuda.synchronize()

    # --- 6. Wipe every trace of the warmup ---------------------------------------
    # (a) main_grad buffers
    for model_chunk in model:
        if hasattr(model_chunk, "zero_grad_buffer"):
            model_chunk.zero_grad_buffer()
    # (b) param.grad fields (backward hooks set grad_added_to_main_grad and clear
    #     .grad, but we belt-and-braces it here in case any codepath left grads)
    for model_chunk in model:
        for p in model_chunk.parameters():
            if p.grad is not None:
                p.grad = None
    if optimizer is not None:
        optimizer.zero_grad()

    # (c) MoE aux-loss accumulators
    try:
        clear_aux_losses_tracker()
    except Exception as e:  # pragma: no cover
        log_rank_0(f"clear_aux_losses_tracker failed (safe to ignore): {e}")

    # (d) Restore all RNG state so training is bit-identical with/without warmup.
    _set_all_rng_states(*saved_rng_states)
    args.check_for_nan_in_loss_and_grad = saved_check_nan
    if hasattr(args, "check_for_spiky_loss"):
        args.check_for_spiky_loss = saved_check_spiky
    if saved_curr_iteration is None:
        if hasattr(args, "curr_iteration"):
            delattr(args, "curr_iteration")
    else:
        args.curr_iteration = saved_curr_iteration

    # (e) Optional: release cached memory to avoid warmup inflating the long-term
    #     watermark (the real iter-1 will re-allocate what it needs).
    if getattr(args, "empty_unused_memory_level", 0) >= 1:
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    t1 = time.time()
    log_rank_0(f"pp warmup wall-time {t1 - t0:.3f}s")


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

    original_train = training.train

    if getattr(original_train, "_primus_pp_warmup_wrapped", False):
        return

    def _train_with_pp_warmup(
        forward_step_func,
        model,
        optimizer,
        opt_param_scheduler,
        train_data_iterator,
        valid_data_iterator,
        process_non_loss_data_func,
        config,
        checkpointing_context,
        non_loss_data_func,
        inference_model=None,
    ):
        args = get_megatron_args()
        if getattr(args, "pp_warmup", False):
            run_pp_warmup(forward_step_func, model, optimizer, config)
        return original_train(
            forward_step_func,
            model,
            optimizer,
            opt_param_scheduler,
            train_data_iterator,
            valid_data_iterator,
            process_non_loss_data_func,
            config,
            checkpointing_context,
            non_loss_data_func,
            inference_model,
        )

    setattr(_train_with_pp_warmup, "_primus_pp_warmup_wrapped", True)
    training.train = _train_with_pp_warmup
    log_rank_0(
        f"[Patch:megatron.pp_warmup] Wrapped train(); PP warmup runs before train when args.pp_warmup = True"
    )
