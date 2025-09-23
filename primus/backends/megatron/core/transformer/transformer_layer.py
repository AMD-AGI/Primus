# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import get_args


def get_transformer_layer_offset(config: TransformerConfig, vp_stage: Optional[int] = None):
    """Get the index offset of current pipeline stage, given the level of pipelining."""
    pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
    if not parallel_state.is_inside_encoder():
        pp_decoder_start = parallel_state.get_pipeline_model_parallel_decoder_start()
        if pp_decoder_start is not None:
            pipeline_rank = pipeline_rank - pp_decoder_start

    if config.pipeline_model_parallel_size > 1:

        if (
            config.num_layers_in_first_pipeline_stage is not None
            or config.num_layers_in_last_pipeline_stage is not None
        ):
            # Calculate number of pipeline stages to distribute the remaining Transformer
            # layers after deducting the Transformer layers in the first or the last stages
            middle_pipeline_stages = config.pipeline_model_parallel_size
            middle_pipeline_stages -= sum(
                [
                    1 if x is not None else 0
                    for x in (
                        config.num_layers_in_first_pipeline_stage,
                        config.num_layers_in_last_pipeline_stage,
                    )
                ]
            )

            # Calculate layers to distribute in each pipeline stage. If the
            # num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage
            # are not set, we will not enable uneven pipeline. All layers will be treated
            # as middle layers.
            num_layers_in_first_pipeline_stage = (
                0
                if config.num_layers_in_first_pipeline_stage is None
                else config.num_layers_in_first_pipeline_stage
            )
            num_layers_in_last_pipeline_stage = (
                0
                if config.num_layers_in_last_pipeline_stage is None
                else config.num_layers_in_last_pipeline_stage
            )

            middle_num_layers = (
                config.num_layers - num_layers_in_first_pipeline_stage - num_layers_in_last_pipeline_stage
            )

            if (vp_size := config.virtual_pipeline_model_parallel_size) is not None:
                assert (
                    vp_stage is not None
                ), "vp_stage must be provided if virtual pipeline model parallel size is set"

                # Calculate number of layers in each virtual model chunk
                # If the num_layers_in_first_pipeline_stage and
                # num_layers_in_last_pipeline_stage are not set, all pipeline stages
                # will be treated as middle pipeline stages in the calculation
                num_layers_per_virtual_model_chunk_in_first_pipeline_stage = (
                    0
                    if config.num_layers_in_first_pipeline_stage is None
                    else config.num_layers_in_first_pipeline_stage // vp_size
                )

                num_layers_per_virtual_model_chunk_in_last_pipeline_stage = (
                    0
                    if config.num_layers_in_last_pipeline_stage is None
                    else config.num_layers_in_last_pipeline_stage // vp_size
                )

                num_layers_per_vritual_model_chunk_in_middle_pipeline_stage = middle_num_layers // vp_size

                # First stage + middle stage + last stage
                total_virtual_chunks = (
                    num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                    + num_layers_per_vritual_model_chunk_in_middle_pipeline_stage
                    + num_layers_per_virtual_model_chunk_in_last_pipeline_stage
                )

                if get_args().patch_zero_bubble and (
                    get_args().zero_bubble_v_schedule or get_args().enable_1f1b_v
                ):
                    assert config.virtual_pipeline_model_parallel_size == 2
                    parallel_state.get_pipeline_model_parallel_world_size()
                    if pipeline_rank == 0:
                        if vp_stage == 0:
                            offset = pipeline_rank
                        else:
                            offset = (
                                total_virtual_chunks
                                - (num_layers_per_virtual_model_chunk_in_last_pipeline_stage)
                                - pipeline_rank
                                - 1
                            )
                else:
                    # Calculate the layer offset with interleaved uneven pipeline parallelism
                    if pipeline_rank == 0:
                        offset = vp_stage * total_virtual_chunks
                    else:
                        offset = (
                            vp_stage * total_virtual_chunks
                            + num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                            + (pipeline_rank - 1)
                            * (
                                num_layers_per_vritual_model_chunk_in_middle_pipeline_stage
                                // middle_pipeline_stages
                            )
                        )
            else:
                if middle_pipeline_stages > 0:
                    num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
                else:
                    num_layers_per_pipeline_rank = 0

                middle_pipeline_rank = (
                    pipeline_rank if config.num_layers_in_first_pipeline_stage is None else pipeline_rank - 1
                )

                if pipeline_rank == 0:
                    offset = 0
                else:
                    offset = (
                        middle_pipeline_rank * num_layers_per_pipeline_rank
                    ) + num_layers_in_first_pipeline_stage
        else:
            num_layers = config.num_layers

            # Increase the number of layers by one if we include the embedding (loss)
            # layer into pipeline parallelism partition and placement
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

            if (vp_size := config.virtual_pipeline_model_parallel_size) is not None:
                assert (
                    vp_stage is not None
                ), "vp_stage must be provided if virtual pipeline model parallel size is set"

                num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
                total_virtual_chunks = num_layers // vp_size
                offset = vp_stage * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

                if get_args().patch_zero_bubble and (
                    get_args().zero_bubble_v_schedule or get_args().enable_1f1b_v
                ):
                    assert config.virtual_pipeline_model_parallel_size == 2
                    if pipeline_rank == 0:
                        if vp_stage == 0:
                            offset = pipeline_rank
                        else:
                            offset = total_virtual_chunks - pipeline_rank - 1
                # Reduce the offset of embedding layer from the total layer number
                if (
                    config.account_for_embedding_in_pipeline_split
                    and not parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
                ):
                    offset -= 1
            else:
                offset = pipeline_rank * num_layers_per_pipeline_rank

                # Reduce the offset of embedding layer from the total layer number
                if (
                    config.account_for_embedding_in_pipeline_split
                    and not parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
                ):
                    offset -= 1
    else:
        offset = 0
    return offset
