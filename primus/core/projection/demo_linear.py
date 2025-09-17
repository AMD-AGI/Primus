###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.projection.meta_module import BaseMetaModule


# ===== Estimation-only LinearProjection =====
class LinearMetaModule(BaseMetaModule):
    """Estimation-only Linear Projection module.
    Provides param/memory/FLOPs estimation, no real tensors involved.
    """

    def __init__(self, input_dim: int, output_dim: int, name="linear_proj"):
        super().__init__(name)
        self.input_dim = input_dim
        self.output_dim = output_dim

    # Parameters
    def estimated_num_params(self) -> int:
        raise NotImplementedError

    # Memory
    def estimated_memory(self, batch_size: int, seq_len: int) -> int:
        raise NotImplementedError

    # Performance
    def estimated_forward_time(self, batch_size: int, seq_len: int) -> int:
        raise NotImplementedError

    def estimated_backward_time(self, batch_size: int, seq_len: int) -> int:
        raise NotImplementedError


# ===== Megatron-specific Linear Projection =====
class MegatronLinearMetaModule(LinearMetaProjection):
    """Megatron implementation of LinearMetaProjection.
    Uses Megatron Tensor Parallel Linear layer to run real forward/backward for measurement.
    """

    def __init__(self, input_dim: int, output_dim: int, tp_size: int = 1, device="cuda", dtype=torch.float16):
        super().__init__(input_dim, output_dim, name="megatron_linear_proj")

    # Parameters
    def measured_num_params(self) -> int:
        raise NotImplementedError

    # Memory
    def measured_memory(self, batch_size: int, seq_len: int) -> int:
        raise NotImplementedError

    # Performance
    def measured_forward_time(self, batch_size: int, seq_len: int) -> int:
        raise NotImplementedError

    def measured_backward_time(self, batch_size: int, seq_len: int) -> int:
        raise NotImplementedError
