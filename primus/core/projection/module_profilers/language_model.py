###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.training_config import TrainingConfig

from .embedding import EmbeddingProfiler
from .layer_norm import LayerNormProfiler
from .loss import LossProfiler
from .output_layer import OutputLayerProfiler
from .transformer_layer import get_dense_transformer_layer_profiler_spec, get_moe_transformer_layer_profiler_spec


def build_profiler(spec: ModuleProfilerSpec, depth=0) -> BaseModuleProfiler:
    """
    Recursively build a profiler instance from a ModuleProfilerSpec.
    """
    if not issubclass(spec.profiler, BaseModuleProfiler):
        raise TypeError(f"spec.profiler must be subclass of BaseModuleProfiler, got {spec.profiler}")

    if depth == 0:
        print(f"Begin build profiler: {spec.profiler.__name__}")

    print(f"{'--'*(depth+1)}[{spec.profiler.__name__}]")

    sub_profilers = {}
    if spec.sub_profiler_specs:
        depth += 1
        for name, sub_spec in spec.sub_profiler_specs.items():
            if sub_spec is None:
                sub_profilers[name] = None
            elif isinstance(sub_spec, ModuleProfilerSpec):
                # build sub profiler with spec
                sub_profilers[name] = build_profiler(sub_spec, depth)
            elif issubclass(sub_spec, BaseModuleProfiler):
                # init sub profile
                print(f"{'--'*(depth+1)}[{sub_spec.__name__}]({name})")
                sub_profilers[name] = sub_spec(spec.config, sub_profilers=None)
            else:
                raise TypeError(f"Invalid type for sub_profiler_specs['{name}']: {type(sub_spec)}")

    return spec.profiler(config=spec.config, sub_profilers=sub_profilers)


def get_language_model_profiler_spec(config: TrainingConfig) -> ModuleProfilerSpec:
    return ModuleProfilerSpec(
        profiler=LanguageModelProfiler,
        config=config,
        sub_profiler_specs={
            "embedding": EmbeddingProfiler,
            "dense_transformer_layer": get_dense_transformer_layer_profiler_spec(config),
            "moe_transformer_layer": get_moe_transformer_layer_profiler_spec(config),
            "final_layernorm": LayerNormProfiler,
            "output_layer": OutputLayerProfiler,
            "calc_loss": LossProfiler,
        },
    )


# language profiler spec -> build_profiler() -> language profiler -> run profiling methods
class LanguageModelProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        rank = int(os.getenv('RANK', '0'))
        self.layers = self.get_layers_for_rank(
            global_rank=rank,
            n_layers=self.config.model_config.num_layers,
            pp_size=self.config.model_parallel_config.pipeline_model_parallel_size,
            tp_size=self.config.model_parallel_config.tensor_model_parallel_size,
            cp_size=self.config.model_parallel_config.context_model_parallel_size,
            ep_size=self.config.model_parallel_config.expert_model_parallel_size,
            num_virtual_pipeline_stages=self.config.model_parallel_config.virtual_pipeline_model_parallel_size)

    def get_layers_for_rank(
        self,
        global_rank: int,
        n_layers: int,
        pp_size: int,
        tp_size: int,
        cp_size: int,
        ep_size: int,
        num_virtual_pipeline_stages: int | None = None
    ) -> list[int]:
        total_stages = pp_size
        if num_virtual_pipeline_stages is not None:
            total_stages = total_stages * num_virtual_pipeline_stages

        if n_layers % total_stages != 0:
            raise ValueError(
                f"Total number of layers ({n_layers}) must be divisible by "
                f"the number of virtual pipeline stages ({total_stages})."
            )

        model_parallel_size = pp_size * tp_size * cp_size * ep_size
        model_parallel_rank = global_rank % model_parallel_size
        pp_rank = model_parallel_rank // (tp_size * cp_size * ep_size)

        # Calculate how many layers are in each virtual stage (chunk)
        layers_per_virtual_stage = n_layers // total_stages

        # A physical pp_rank hosts multiple virtual stages in an interleaved fashion.
        # pp_rank 0 gets virtual stages: 0, pp_size, 2*pp_size, ...
        # pp_rank 1 gets virtual stages: 1, pp_size+1, 2*pp_size+1, ...
        my_virtual_stages = range(pp_rank, total_stages, pp_size)

        assigned_layers = []
        for vs_index in my_virtual_stages:
            start_layer = vs_index * layers_per_virtual_stage
            end_layer = (vs_index + 1) * layers_per_virtual_stage - 1
            for layer in range(start_layer, end_layer + 1):
                assigned_layers.append(layer)

        return assigned_layers

    def get_dp_size(self) -> int:
        num_nodes = int(os.getenv('NNODES', '1'))
        if num_nodes == 1:
            # Calculate the minimum number of needed nodes
            num_nodes = (self.config.model_parallel_config.tensor_model_parallel_size *
                self.config.model_parallel_config.context_model_parallel_size *
                self.config.model_parallel_config.pipeline_model_parallel_size *
                self.config.model_parallel_config.expert_model_parallel_size //
                int(os.getenv('GPUS_PER_NODE', '8')))
        world_size = num_nodes * int(os.getenv('GPUS_PER_NODE', '8'))
        dp_size = (world_size //
                self.config.model_parallel_config.expert_model_parallel_size //
                self.config.model_parallel_config.pipeline_model_parallel_size)
        return dp_size

    def get_num_bytes_per_param(self) -> float:
        dp_size = self.get_dp_size()
        multiplier = 4  # param weights + gradients, bf16
        # 2 for main params, 4 + 4 for fp32 optimizer 1st & 2nd order moments
        optimizer_state_multiplier = 10 / dp_size  # DP sharding
        return multiplier + optimizer_state_multiplier

    def estimated_num_params(self, rank: int | None = None) -> int:
        total_params = 0
        if rank is None:
            layers = range(self.config.model_config.num_layers)
        else:
            layers = self.layers
        for layer in layers:
            is_moe = self.config.model_config.moe_pattern[layer]
            if is_moe:
                total_params += self.sub_profilers["moe_transformer_layer"].estimated_num_params(rank)
            else:
                total_params += self.sub_profilers["dense_transformer_layer"].estimated_num_params(rank)
        if 0 in self.layers:
            total_params += self.sub_profilers["embedding"].estimated_num_params(rank)
        if self.config.model_config.num_layers - 1 in self.layers:
            total_params += self.sub_profilers["final_layernorm"].estimated_num_params(rank)
            total_params += self.sub_profilers["output_layer"].estimated_num_params(rank)
            total_params += self.sub_profilers["calc_loss"].estimated_num_params(rank)
        return total_params

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        total_act = 0
        pp_size = self.config.model_parallel_config.pipeline_model_parallel_size
        vpp_size = self.config.model_parallel_config.virtual_pipeline_model_parallel_size
        for layer in self.layers:
            is_moe = self.config.model_config.moe_pattern[layer]
            if is_moe:
                total_act += self.sub_profilers["moe_transformer_layer"].estimated_activation_memory(batch_size, seq_len)
            else:
                total_act += self.sub_profilers["dense_transformer_layer"].estimated_activation_memory(batch_size, seq_len)
        if 0 in self.layers:
            total_act += self.sub_profilers["embedding"].estimated_activation_memory(batch_size, seq_len)
        if self.config.model_config.num_layers - 1 in self.layers:
            total_act += self.sub_profilers["final_layernorm"].estimated_activation_memory(batch_size, seq_len)
            total_act += self.sub_profilers["output_layer"].estimated_activation_memory(batch_size, seq_len)
            total_act += self.sub_profilers["calc_loss"].estimated_activation_memory(batch_size, seq_len)
        # 1F1B
        total_act *= pp_size
        interleaved_schedule_memory_penalty = 1 + (
            (pp_size - 1)
            / (pp_size * vpp_size)
        )
        ga = self.config.runtime_config.global_batch_size // self.get_dp_size()
        gs_saving = 1 if ga > pp_size else ga / pp_size
        total_act *= gs_saving * interleaved_schedule_memory_penalty
        return total_act
