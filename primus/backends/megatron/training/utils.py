# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""
import torch
from megatron.core import mpu, parallel_state
from megatron.training import get_args


def is_second_last_pipeline_stage():
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_rank_to_run = pipeline_model_parallel_size - 2 if pipeline_model_parallel_size > 1 else 0
    return (
        parallel_state.get_pipeline_model_parallel_rank() == pipeline_rank_to_run
        and parallel_state.get_data_parallel_rank() == 0
        and parallel_state.get_tensor_model_parallel_rank() == 0
    )


def print_second_last_pipeline_stage(message):
    if torch.distributed.is_initialized():
        if is_second_last_pipeline_stage():
            print(message, flush=True)
    else:
        print(message, flush=True)


def is_pipeline_stage_containing_loss():
    if get_args().zero_bubble_v_schedule or get_args().enable_1f1b_v:
        return mpu.is_pipeline_first_stage(ignore_virtual=True)
    else:
        return mpu.is_pipeline_last_stage(ignore_virtual=True)
