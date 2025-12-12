###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

__all__ = [
    "megatron_primuspipe_handler_dict",
]

from primus.core.pipeline_parallel.handler.wgrad_handler import default_wgrad_handler
from primus.core.pipeline_parallel.scheduler.scheduler_node import FuncType

from .bwd_handler import megatron_bwd_handler
from .communication_handler import batch_p2p_communication_handler
from .fwd_handler import megatron_fwd_handler

megatron_primuspipe_handler_dict = {
    FuncType.F: megatron_fwd_handler,
    FuncType.B: megatron_bwd_handler,
    FuncType.W: default_wgrad_handler,
    FuncType.BW: megatron_bwd_handler,
    FuncType.SF: batch_p2p_communication_handler,
    FuncType.SB: batch_p2p_communication_handler,
    FuncType.RF: batch_p2p_communication_handler,
    FuncType.RB: batch_p2p_communication_handler,
}
