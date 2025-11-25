from dataclasses import dataclass
from enum import Enum
from typing import Any


class FuncType(Enum):
    F = "FORWARD"
    B = "BACKWARD"
    W = "W_GRAD"
    BW = "BACKWARD_W_GRAD"
    FB = "FORWARD_BACKWARD"
    SF = "SEND_FORWARD"
    SB = "SEND_BACKWARD"
    RF = "RECV_FORWARD"
    RB = "RECV_BACKWARD"


@dataclass
class SchedulerNode:
    """Schedule Node
    SchedulerNode is a node in the scheduler pipeline which will be executed by the handler.
    Each kind of FuncType should privide some specific arguments for the handler.
    """

    func_type: FuncType
    mini_batch: int
    chunk: int
    args: dict[str, Any] = None
    meta: dict[str, Any] = None

    def __str__(self):
        return f"({self.func_type.name}|{self.mini_batch}|{self.chunk})"
