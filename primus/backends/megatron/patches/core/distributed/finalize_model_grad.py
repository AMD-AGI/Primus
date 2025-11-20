###############################################################################
# Some parts of this code are copied and modified from
# Sea AI Lab's zero-bubble-pipeline-parallelism project
# (https://github.com/sail-sg/zero-bubble-pipeline-parallelism).
#
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

from typing import List, Optional

import torch
from megatron.core import parallel_state
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_layernorm_grads,
    _allreduce_word_embedding_grads,
)
from megatron.core.utils import get_model_config


def finalize_model_grads(model: List[torch.nn.Module], num_tokens: Optional[torch.Tensor] = None):
    """
    All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied),
    scale gradients by `num_tokens`.
    """

    config = get_model_config(model[0])

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:
        config.timers("all-grads-sync", log_level=1).start(barrier=config.barrier_with_L1_time)
    for model_chunk in model:
        model_chunk.finish_grad_sync()
    if config.timers is not None:
        config.timers("all-grads-sync").stop()

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:
        config.timers("layernorm-grads-all-reduce", log_level=1).start(barrier=config.barrier_with_L1_time)
    _allreduce_layernorm_grads(model, config)
    if config.timers is not None:
        config.timers("layernorm-grads-all-reduce").stop()

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:
        config.timers("embedding-grads-all-reduce", log_level=1).start(barrier=config.barrier_with_L1_time)
    _allreduce_word_embedding_grads(model, config)
    if config.timers is not None:
        config.timers("embedding-grads-all-reduce").stop()

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:
        # the number of tokens is only present on the last stage, so broadcast it
        # to the other ranks in the pipeline parallel group.
        from megatron.training import get_args

        last_layer_rank = parallel_state.get_pipeline_model_parallel_last_rank()
        if get_args().zero_bubble_v_schedule or get_args().enable_1f1b_v:
            last_layer_rank = parallel_state.get_pipeline_model_parallel_first_rank()
        torch.distributed.broadcast(
            num_tokens,
            src=last_layer_rank,
            group=parallel_state.get_pipeline_model_parallel_group(),
        )
        # all-reduce across DP ranks.
        torch.distributed.all_reduce(num_tokens, group=parallel_state.get_data_parallel_group())
        for model_chunk in model:
            if num_tokens > 0:
                scaling = 1.0 / num_tokens
                model_chunk.scale_gradients(scaling)
