###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from .trainer import MegatronTrainer


class MegatronSFTTrainer(MegatronTrainer):
    def __init__(self, *args, **kwargs):
        kwargs["module_name"] = "sft_trainer"
        super().__init__(*args, **kwargs)

    def get_batch(self, data_iterator):
        raise NotImplementedError

    def loss_func(self, loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        raise NotImplementedError

    def forward_step(self, data_iterator, model: GPTModel):
        raise NotImplementedError
