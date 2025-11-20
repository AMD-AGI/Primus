###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import List, Optional

import torch
from megatron.core import tensor_parallel
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.distributed.torch_fully_sharded_data_parallel import (
    TorchFullyShardedDataParallel,
)
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer

from primus.modules.module_utils import warning_rank_0


class PrimusTorchFullyShardedDataParallel(TorchFullyShardedDataParallel):
    """
    Customized FSDP implementation for Primus framework, with pre-defined submodules to wrap.
    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        sub_modules_to_wrap: Optional[List[torch.nn.Module]] = None,
        **kwargs,
    ):
        if sub_modules_to_wrap is None:
            sub_modules_to_wrap = [
                TransformerLayer,
                LanguageModelEmbedding,
                RotaryEmbedding,
                tensor_parallel.ColumnParallelLinear,
            ]

        if kwargs:
            warning_rank_0(f"PrimusTorchFullyShardedDataParallel: not use args: {kwargs}")

        super().__init__(
            config=config, ddp_config=ddp_config, module=module, sub_modules_to_wrap=sub_modules_to_wrap
        )
