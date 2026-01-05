###############################################################################
# Some parts of this code are copied and modified from
# Sea AI Lab's zero-bubble-pipeline-parallelism project
# (https://github.com/sail-sg/zero-bubble-pipeline-parallelism).
#
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import torch
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_virtual_pipeline_model_parallel_rank,
    get_virtual_pipeline_model_parallel_world_size,
    is_pipeline_first_stage,
)

from primus.modules.trainer.megatron.utils import is_v_schedule_enabled


def default_embedding_ranks(pp_ranks, split_rank=None):
    """Return the default ranks that constitute the stages on which the word embeddings live.
    For most models, these are the first and last pipeline stages.

    We also support the deprecated split rank argument for backwards compatibility."""

    if len(pp_ranks) == 1:
        return [pp_ranks[0]]
    elif split_rank is not None and pp_ranks[split_rank] not in (pp_ranks[0], pp_ranks[-1]):
        assert not is_v_schedule_enabled()
        return [pp_ranks[0], pp_ranks[split_rank], pp_ranks[-1]]
    else:
        if is_v_schedule_enabled():
            return [pp_ranks[0]]
        return [pp_ranks[0], pp_ranks[-1]]


def is_pipeline_last_stage(ignore_virtual=False, vp_stage=None):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""

    if not ignore_virtual and get_virtual_pipeline_model_parallel_world_size() is not None:

        if vp_stage is None:
            vp_stage = get_virtual_pipeline_model_parallel_rank()

        virtual_pipeline_model_parallel_world_size = get_virtual_pipeline_model_parallel_world_size()
        if is_v_schedule_enabled():
            assert virtual_pipeline_model_parallel_world_size == 2
            return get_pipeline_model_parallel_rank() == 0 and vp_stage == (
                get_virtual_pipeline_model_parallel_world_size() - 1
            )

        # if vp_stage == (get_virtual_pipeline_model_parallel_world_size() - 1):
        #     assert get_pipeline_model_parallel_rank() == 0

        if vp_stage != (get_virtual_pipeline_model_parallel_world_size() - 1):
            return False

    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    from megatron.core.parallel_state import _EMBEDDING_GLOBAL_RANKS

    global _EMBEDDING_GLOBAL_RANKS
    if _EMBEDDING_GLOBAL_RANKS is None:
        return False
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if is_v_schedule_enabled():
        return is_pipeline_first_stage(ignore_virtual=False) or is_pipeline_last_stage(ignore_virtual=False)
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False
