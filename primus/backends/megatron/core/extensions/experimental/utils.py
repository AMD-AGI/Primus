import torch
from typing import Callable

def init_method_constant(val: float) -> Callable:
    """Init method to set all tensor elements to a constant value."""
    if val == 1.0:

        def init_(tensor: torch.Tensor) -> Callable:
            return torch.nn.init.ones_(tensor)

    elif val == 0.0:

        def init_(tensor: torch.Tensor) -> Callable:
            return torch.nn.init.zeros_(tensor)

    else:

        def init_(tensor: torch.Tensor) -> Callable:
            return torch.nn.init.constant_(tensor, val)

    return init_

def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} is not divisible by {denominator}")


def divide(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator