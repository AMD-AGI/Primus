from primus.core.projection.pipeline_simulation.scheduler.scheduler_node import FuncType

from .bwd_handler import default_bwd_handler
from .bwd_wgrad_handler import default_bwd_wgrad_handler
from .communication_handler import batch_p2p_communication_handler
from .fwd_handler import default_fwd_handler
from .wgrad_handler import default_wgrad_handler

default_handler_dict = {
    FuncType.F: default_fwd_handler,
    FuncType.B: default_bwd_handler,
    FuncType.W: default_wgrad_handler,
    FuncType.BW: default_bwd_wgrad_handler,
    FuncType.SF: batch_p2p_communication_handler,
    FuncType.SB: batch_p2p_communication_handler,
    FuncType.RF: batch_p2p_communication_handler,
    FuncType.RB: batch_p2p_communication_handler,
}
