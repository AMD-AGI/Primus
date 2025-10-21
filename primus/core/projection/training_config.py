###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Configuration for training the profiler models.
    """

    model_config: ModelConfig
    runtime_config: RuntimeConfig
    model_parallel_config: ModelParallelConfig


@dataclass
class RuntimeConfig:
    global_batch_size: int = 1
    micro_batch_size: int = 1
    sequence_length: int = 0


@dataclass
class ModelParallelConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int = 1
    context_model_parallel_size: int = 1
    torch_fsdp2: bool = False
    distributed_optimizer: bool = False


@dataclass
class ModelConfig:
    num_layers: int = 0
    hidden_size: int = 0
