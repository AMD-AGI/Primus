###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import time

import torch
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
from megatron.training import ft_integration, get_args, get_timers
from megatron.training.utils import is_last_rank

from primus.backends.megatron.training.global_vars import get_train_start_time
from primus.backends.megatron.training.utils import is_pipeline_stage_containing_loss
from primus.modules.module_utils import log_rank_0


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

    total_loss_dict = {}

    # make validation batch size independent from training batch size
    eval_batch_size = args.global_batch_size
    eval_num_microbatches = eval_batch_size // (args.micro_batch_size * args.data_parallel_size)
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
                # Average loss across microbatches.
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
                        else:
                            # legacy behavior. we average over the number of microbatches,
                            # and so the denominator is 1.
                            numerator += val
                            denominator += 1
                    total_loss_dict[key] = numerator / denominator

            args.consumed_valid_samples += eval_batch_size

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

    rerun_state_machine.set_mode(rerun_mode)

    return total_loss_dict, collected_non_loss_data, False
