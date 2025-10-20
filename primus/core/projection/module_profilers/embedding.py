###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.projection.base_module_profiler import BaseModuleProfiler


class EmbeddingProfiler(BaseModuleProfiler):
    def __init__(self, name: str):
        self.name = name

    # -------- Parameter related --------
    @abstractmethod
    def estimated_num_params(self) -> int:
        # embedding + layers + outputlayer
        return 0

    @abstractmethod
    def measured_num_params(self) -> int:
        return 0

    # -------- Memory related --------
    @abstractmethod
    def estimated_memory(self, batch_size: int, seq_len: int) -> int:
        return 0

    @abstractmethod
    def measured_memory(self, batch_size: int, seq_len: int) -> int:
        return 0

    # -------- Performance related --------
    @abstractmethod
    def estimated_forward_time(self, batch_size: int, seq_len: int) -> int:
        return 0

    @abstractmethod
    def estimated_backward_time(self, batch_size: int, seq_len: int) -> int:
        return 0

    @abstractmethod
    def measured_forward_time(self, batch_size: int, seq_len: int) -> float:
        return 0

    @abstractmethod
    def measured_backward_time(self, batch_size: int, seq_len: int) -> float:
        return 0
