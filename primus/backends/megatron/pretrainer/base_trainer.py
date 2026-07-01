###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from abc import ABC, abstractmethod

import torch
from megatron.core.models.gpt import GPTModel


class BaseTrainer(ABC):
    @abstractmethod
    def get_batch(self, data_iterator):
        pass

    @abstractmethod
    def loss_func(self, loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        pass

    @abstractmethod
    def forward_step(self, data_iterator, model: GPTModel):
        pass
