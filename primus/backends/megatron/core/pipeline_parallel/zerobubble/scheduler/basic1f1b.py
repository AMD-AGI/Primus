###############################################################################
# Some parts of this code are copied and modified from
# Sea AI Lab's zero-bubble-pipeline-parallelism project
# (https://github.com/sail-sg/zero-bubble-pipeline-parallelism).
#
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

from primus.modules.module_utils import log_rank_0

from .graph import BW, F, GraphConfig, ScheduledNode


def create_schedule(config: GraphConfig):
    local_order = []
    for stage in range(config.n_stages):
        order = []
        num_warmup = min(config.n_stages - stage - 1, config.n_micro)
        num_remaining = config.n_micro - num_warmup
        funcs = []
        for mb in range(num_warmup):
            funcs.append((F, mb))
        for mb in range(num_remaining):
            funcs.append((F, num_warmup + mb))
            funcs.append((BW, mb))
        for i in range(num_warmup):
            funcs.append((BW, num_remaining + i))

        log_rank_0(" ".join([f"{t.value}{mb}" for (t, mb) in funcs]))

        for func_type, mb in funcs:
            order.append(
                ScheduledNode(
                    type=func_type,
                    stage=stage,
                    microbatch=mb,
                    layer_group_idx=stage,
                )
            )
        local_order.append(order)
    return local_order
