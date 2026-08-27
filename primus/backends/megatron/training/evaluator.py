###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import time

import torch
from megatron.core import parallel_state
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
from megatron.training import ft_integration, get_args, get_timers
from megatron.training.utils import is_last_rank

from primus.backends.megatron.training.eval_budget import get_eval_num_microbatches
from primus.backends.megatron.training.global_vars import get_train_start_time
from primus.backends.megatron.training.utils import is_pipeline_stage_containing_loss
from primus.core.utils.module_utils import log_rank_0

# The key under which the diffusion validation path reports
# (summed per-sample loss, sample count). Its denominator is the only one that
# is a sample count rather than a microbatch count.
VAL_LOSS_KEY = "loss"


def _record_consumed_valid_samples(args, observed_samples, eval_iters, eval_batch_size):
    """Account for the samples the evaluation actually read, and say so.

    The previous behaviour added ``eval_batch_size`` per iteration regardless
    of how wide the batches really were, so a short final batch on any worker
    was counted as a full one and the logged sample count could exceed the
    count evaluated. Reporting the intended number is worse than reporting
    nothing, because it is indistinguishable from a correct run.
    """
    expected = eval_iters * eval_batch_size

    if observed_samples is None:
        # No loss-bearing stage on this rank, or a metric shape that carries
        # microbatch counts rather than sample counts. Fall back to the
        # configured budget rather than skipping the accounting entirely.
        args.consumed_valid_samples += expected
        return

    args.consumed_valid_samples += observed_samples

    if observed_samples != expected:
        # Context parallelism duplicates the per-sample loss across CP ranks,
        # which inflates the reduced denominator; only assert when it cannot.
        cp_size = parallel_state.get_context_parallel_world_size()
        detail = (
            f"Evaluation read {observed_samples} samples but the configuration "
            f"implies {expected} ({eval_iters} iterations x {eval_batch_size}). "
            f"Difference: {expected - observed_samples}."
        )
        if cp_size == 1:
            raise RuntimeError(
                f"{detail}\nThis is the silent under-read described in eval_budget: "
                f"Energon workers whose batch quota is short leave the tail of their "
                f"slice unread. Check val_num_workers against eval_samples."
            )
        log_rank_0(f"[eval] {detail} (context_parallel_size={cp_size}, not asserting)")


def _reduction_device(numerators):
    """Where to build the packed buffer: wherever the accumulators already live.

    Packing onto the accumulators' own device avoids a transfer and keeps the
    buffer on the device the process group can reduce over.
    """
    for value in numerators.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cuda")


def reduce_eval_losses(numerators, denominators, dp_group):
    """Reduce every accumulated (numerator, denominator) pair across data parallelism.

    One reduction, one group, one host sync.

    Every numerator and denominator across every key is packed into a single
    fp64 buffer and reduced once. The reduction must produce a value identical
    on every rank: the target-eval-loss early stop compares it against a
    threshold, and if ranks disagree near the target one can exit train() alone
    while the others keep training, desyncing collectives (grad-norm
    all-reduce) into an NCCL hang.

    The previous implementation reduced num/den over DP-with-CP and then
    reduced the result again over DP-without-CP. That left both sides
    multiplied by data_parallel_size, so the ratio was right but the
    denominator could not be read as a sample count.

    Returns:
        ``(total_loss_dict, observed_samples)``, where ``observed_samples`` is
        the globally reduced ``VAL_LOSS_KEY`` denominator -- a true sample
        count -- or None when no such key was reported.
    """
    keys = sorted(numerators.keys())
    packed = torch.tensor(
        [float(value) for key in keys for value in (numerators[key], denominators[key])],
        dtype=torch.float64,
        device=_reduction_device(numerators),
    )
    torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM, group=dp_group)

    # Single host sync for every key at once.
    reduced = packed.tolist()
    total_loss_dict = {}
    observed_samples = None
    for index, key in enumerate(keys):
        numerator, denominator = reduced[2 * index], reduced[2 * index + 1]
        # Keep the result as a 0-dim tensor: downstream Megatron code
        # (evaluate_and_print_results) and mlperf logging call .item().
        if denominator > 0:
            total_loss_dict[key] = torch.tensor(
                numerator / denominator, dtype=torch.float32, device=packed.device
            )
        else:
            total_loss_dict[key] = torch.zeros((), dtype=torch.float32, device=packed.device)
        if key == VAL_LOSS_KEY:
            observed_samples = int(round(denominator))

    return total_loss_dict, observed_samples


def primus_evaluate(
    forward_step_func,
    data_iterator,
    model,
    process_non_loss_data_func,
    config,
    verbose=True,
    non_loss_data_func=None,
    eval_iters=None,
):
    """Evaluation."""
    args = get_args()
    timers = get_timers()

    timers("evaluate", log_level=0).start(barrier=True)

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        from megatron.legacy.model.vision.knn_monitor import compute_feature_bank

        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    # Disable result validation during evaluation
    rerun_state_machine = get_rerun_state_machine()
    rerun_mode = rerun_state_machine.get_mode()
    rerun_state_machine.set_mode(RerunMode.DISABLED)

    # Accumulate numerator and denominator separately across all eval iterations
    total_loss_numerators = {}
    total_loss_denominators = {}

    # make validation batch size independent from training batch size
    eval_batch_size = args.global_batch_size
    # Shared with the dataloader provider so the loop and the dataset it reads
    # from cannot disagree about how large an evaluation is.
    eval_num_microbatches = get_eval_num_microbatches(args)
    forward_backward_func = get_forward_backward_func()
    if args.enable_cuda_graph and args.cuda_graph_scope == "full_iteration":
        forward_backward_func = FullCudaGraphWrapper(
            forward_backward_func, cuda_graph_warmup_steps=args.cuda_graph_warmup_steps
        )

    if eval_iters is None:
        eval_iters = args.eval_iters

    with torch.no_grad():
        iteration = 0
        if verbose:
            log_rank_0(f"Evaluating on {eval_iters * eval_batch_size} samples")
        while iteration < eval_iters:
            iteration += 1
            if verbose:
                log_rank_0(f"Evaluating iter {iteration}/{eval_iters}")

            # Don't care about timing during evaluation
            config.timers = None
            ft_integration.on_eval_step_start()
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
            )
            ft_integration.on_eval_step_end()
            config.timers = get_timers()

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if is_pipeline_stage_containing_loss():
                # Accumulate loss across microbatches for this iteration.
                for key in loss_dicts[0].keys():
                    numerator = 0
                    denominator = 0
                    for x in loss_dicts:
                        val = x[key]
                        # there is one dict per microbatch. in new reporting, we average
                        # over the total number of tokens across the global batch.
                        if isinstance(val, tuple) or isinstance(val, list):
                            numerator += val[0]
                            denominator += val[1]
                        elif isinstance(val, torch.Tensor) and val.numel() == 2:
                            # [loss, num_tokens] from pretrain_gpt loss_func (Megatron default)
                            numerator += val[0]
                            denominator += val[1]
                        else:
                            # legacy behavior. we average over the number of microbatches,
                            # and so the denominator is 1.
                            numerator += val
                            denominator += 1
                    # Accumulate across all eval iterations
                    if key not in total_loss_numerators:
                        total_loss_numerators[key] = 0
                        total_loss_denominators[key] = 0
                    total_loss_numerators[key] += numerator
                    total_loss_denominators[key] += denominator

            if args.exit_duration_in_mins:
                train_time = (time.time() - get_train_start_time()) / 60.0
                done_cuda = torch.tensor(
                    [train_time > args.exit_duration_in_mins], dtype=torch.int, device="cuda"
                )
                torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    rerun_state_machine.set_mode(rerun_mode)
                    log_rank_0("Exiting during evaluation, timelimit reached")
                    return None, None, True

        total_loss_dict = {}
        observed_samples = None
        if is_pipeline_stage_containing_loss():
            from megatron.core import mpu

            total_loss_dict, observed_samples = reduce_eval_losses(
                total_loss_numerators,
                total_loss_denominators,
                mpu.get_data_parallel_group(with_context_parallel=True),
            )

        _record_consumed_valid_samples(args, observed_samples, eval_iters, eval_batch_size)

        collected_non_loss_data = None
        if non_loss_data_func is not None:
            collected_non_loss_data = non_loss_data_func(model)
        elif process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                collect_non_loss_data=True,
            )

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    timers("evaluate").stop()
    timers.log(["evaluate"])

    rerun_state_machine.set_mode(rerun_mode)

    return total_loss_dict, collected_non_loss_data, False
