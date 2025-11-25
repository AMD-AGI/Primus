__all__ = [
    "megatron_primuspipe_handler_dict",
]

from primus.core.pipeline_parallel.handler.wgrad_handler import default_wgrad_handler
from primus.core.pipeline_parallel.scheduler.scheduler_node import FuncType
from .communication_handler import batch_p2p_communication_handler
from .fwd_handler import megatron_fwd_handler
from .bwd_handler import megatron_bwd_handler

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