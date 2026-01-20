###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from megatron.training import get_args

from primus.backends.megatron.core.pipeline_parallel.zerobubble.zbpp_utils import (
    WeightGradStore,
)
from primus.core.pipeline_parallel.handler.wgrad_handler import WGradRunningCache


def insert_wgrad_func_into_cache(weight, preprocess_func, process_wgrad_func):

    def wgrad_func():
        grad_output, input, _ = preprocess_func()
        process_wgrad_func(grad_output, input, None)

    if get_args().patch_zero_bubble:
        assert not get_args().patch_primus_pipeline, "Cannot patch both zero bubble and primus pipeline"
        if WeightGradStore.split_bw():
            WeightGradStore.put(
                weight,
                preprocess_func,
                process_wgrad_func,
            )
        else:
            wgrad_func()
    elif get_args().patch_primus_pipeline:
        WGradRunningCache.append(wgrad_func)
    else:
        raise ValueError("Invalid patch mode")
