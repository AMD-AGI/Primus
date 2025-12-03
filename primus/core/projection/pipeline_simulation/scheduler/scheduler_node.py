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


@dataclass(eq=False)
class SchedulerNode:
    """Schedule Node
    SchedulerNode is a node in the scheduler pipeline which will be executed by the handler.
    Each kind of FuncType should privide some specific arguments for the handler.

    Examples:
        F: {
            args: {
                fwd_func: function(x: Optional[List[Tensor]|None], minibatch: int, chunk: int) -> List[Tensor]
                output: List[Tensor]
            }
        }

        B/BW: {
            args: {
                backward_func: function(input_tensors, output_tensors, output_grad, minibatch: int, chunk: int) -> List[Tensor]
                output: List[Tensor]
                output_grad: List[Tensor]
            }
        }

        W: {
            args: {
                w_func: function(minibatch: int, chunk: int) -> None
            }
        }

        SF/SB: {
            args: {} # no arguments, schedulerwill auto find output of previous node as input
        }

        RF/RB: {
            args: { # hold the buffer
                fwd_input/bwd_output: List[Tensor]
            }
        }
    """

    func_type: FuncType
    mini_batch: int
    chunk: int
    args: Any = None
    meta: Any = None

    def __str__(self):
        return f"({self.func_type.name}|{self.mini_batch}|{self.chunk})"

    def __eq__(self, other):
        if not isinstance(other, SchedulerNode):
            return False
        return (
            self.func_type == other.func_type
            and self.mini_batch == other.mini_batch
            and self.chunk == other.chunk
        )

    def __hash__(self):
        return hash((self.func_type, self.mini_batch, self.chunk))
