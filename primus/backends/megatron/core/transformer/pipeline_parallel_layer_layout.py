import copy
from itertools import chain

from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.pipeline_parallel_layer_layout import (
    PipelineParallelLayerLayout,
)

from primus.modules.trainer.megatron.utils import is_v_schedule_enabled


class PrimusPipelineParallelLayerLayout(PipelineParallelLayerLayout):
    def __init__(self, layout: str | list, pipeline_model_parallel_size: int):
        if not is_v_schedule_enabled():
            super().__init__(layout, pipeline_model_parallel_size)
        else:
            self.input_data = layout
            if isinstance(layout, str):
                layout = PipelineParallelLayerLayout.parse_str_to_list(layout)
            else:
                layout = copy.deepcopy(layout)
            assert all(isinstance(row, list) for row in layout), (
                f"pipeline_model_parallel_layout must be a list of lists, but got"
                f" {[type(row) for row in layout]=}"
            )

            # Check PP size and get VPP size
            assert len(layout) % pipeline_model_parallel_size == 0, (
                f"pipeline_model_parallel_layout must be divisible"
                f" by pipeline_model_parallel_size ({len(layout)=},"
                f" {pipeline_model_parallel_size=})"
            )
            virtual_pipeline_model_parallel_size = len(layout) // pipeline_model_parallel_size

            assert virtual_pipeline_model_parallel_size == 2, "v-scheduled must have vpp size of 2"

            # Convert 1D layout to 2D layout
            layout = [
                [layout[pp_rank], layout[pipeline_model_parallel_size * 2 - 1 - pp_rank]]
                for pp_rank in range(pipeline_model_parallel_size)
            ]

            # Convert all strings in pipeline_model_parallel_layout to LayerType
            for pp_rank in range(pipeline_model_parallel_size):
                for vpp_rank in range(virtual_pipeline_model_parallel_size):
                    transferred_layout = []
                    for layer_type in layout[pp_rank][vpp_rank]:
                        assert isinstance(layer_type, LayerType) or isinstance(layer_type, str), (
                            f"elements in pipeline_model_parallel_layout must be LayerType or str,"
                            f" but got {type(layer_type)}."
                        )
                        if isinstance(layer_type, str):
                            layer_type = layer_type.strip().lower()
                            assert (
                                layer_type in LayerType.__members__
                            ), f"{layer_type} is not a valid LayerType"
                            layer_type = LayerType[layer_type]
                        transferred_layout.append(layer_type)
                    layout[pp_rank][vpp_rank] = transferred_layout

            # Flatten the pipeline layout in layer id order.
            nested_layout = [[] for _ in range(pipeline_model_parallel_size * 2)]
            for pp_rank in range(pipeline_model_parallel_size):
                nested_layout[pp_rank] = layout[pp_rank][0]
                nested_layout[pipeline_model_parallel_size * 2 - 1 - pp_rank] = layout[pp_rank][1]

            flatten_layout = list(chain.from_iterable(nested_layout))

            self.pipeline_model_parallel_size = pipeline_model_parallel_size
            self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
            self.layout = layout
            self.flatten_layout = flatten_layout
