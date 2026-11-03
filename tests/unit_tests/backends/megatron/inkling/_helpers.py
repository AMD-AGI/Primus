###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared fixtures for the Inkling unit tests: a gloo process group (rendezvous through a
FileStore, so no TCP port is needed) with Megatron's model-parallel state, and a small
Inkling config."""

import tempfile

import pytest
import torch


def init_parallel(store_path=None, rank=0, world_size=1, expert_parallel_size=1, tensor_parallel_size=1):
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is not available")
    if not torch.distributed.is_initialized():
        if store_path is None:
            store_path = tempfile.mkstemp(prefix="inkling-ut-store-")[1]
        store = torch.distributed.FileStore(store_path, world_size)
        torch.distributed.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_parallel_size, 1, expert_model_parallel_size=expert_parallel_size
        )
    try:
        model_parallel_cuda_manual_seed(123)
    except Exception:  # no CUDA: the tracker is only used for (zero) attention dropout
        pass


def small_config(**overrides):
    from primus.backends.megatron.core.models.inkling.inkling_transformer_config import (
        InklingTransformerConfig,
    )

    kw = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=16,
        ffn_hidden_size=32,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_ffn_hidden_size=32,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_score_function="sigmoid",
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        init_method_std=0.02,
        inkling_d_rel=4,
        inkling_rel_extent=32,
        inkling_sliding_window=8,
        inkling_local_layer_ids=[0],
        inkling_swa_num_attention_heads=4,
        inkling_swa_num_query_groups=4,
        inkling_swa_kv_channels=16,
        inkling_log_scaling_n_floor=4.0,
        inkling_log_scaling_alpha=0.1,
        inkling_num_shared_experts=2,
        inkling_unpadded_vocab_size=90,
    )
    kw.update(overrides)
    return InklingTransformerConfig(**kw)


def destroy_parallel():
    """Undo init_parallel: Megatron's model-parallel state and the process group."""
    from megatron.core import parallel_state

    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.fixture(scope="module")
def parallel():
    """The parallel state for one test module, torn down after it (when this fixture set it
    up), so a later module does not see a leaked process group."""
    from megatron.core import parallel_state

    owned = not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        or parallel_state.model_parallel_is_initialized()
    )
    init_parallel()
    yield
    if owned:
        destroy_parallel()
