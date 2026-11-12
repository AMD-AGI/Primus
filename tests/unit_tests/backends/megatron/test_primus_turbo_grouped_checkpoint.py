###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``torch_dist`` checkpoints of ``PrimusTurboGroupedLinear``, the routed experts of every
Megatron MoE model run with ``use_turbo_grouped_gemm``.

Turbo trains each expert linear as one stacked ``weights`` tensor. Its checkpoint entries must
be TE's per-expert ones, key for key, so checkpoints move between Turbo and TE runs and between
expert-parallel sizes, and the distributed optimizer must find ``weights`` to save its state.
"""

import re
import socket
from types import SimpleNamespace

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()
pytest.importorskip("primus_turbo")

NUM_EXPERTS = 4
FULLY_RESHARDABLE = {"distrib_optim_sharding_type": "fully_reshardable"}

# The Primus fields Turbo's layers read from Megatron's global args.
GLOBAL_ARGS = dict(
    enable_turbo_attention_float8=False,
    data_parallel_replicate_degree=1,
    offload=False,
    offload_ops=[],
    patch_primus_pipeline=False,
    pp_algorithm=None,
    patch_zero_bubble=False,
    enable_zero_bubble=False,
    use_turbo_fused_act_with_probs=False,
    moe_router_padding_for_quantization=False,
)


def _set_experts(turbo: bool, te_provider) -> None:
    """Build GPT layer specs with TE's spec provider, or with Turbo's and its grouped GEMM on,
    as the ``megatron.turbo.te_spec_provider`` patch does."""
    from megatron.core.models.gpt import gpt_layer_specs

    if not turbo:
        gpt_layer_specs.TESpecProvider = te_provider
        return
    from primus.backends.megatron.core.extensions import (
        transformer_engine_spec_provider as spec_provider,
    )

    cfg = SimpleNamespace(
        **{
            **vars(spec_provider._build_default_primus_args()),
            **dict(enable_primus_turbo=True, use_turbo_grouped_gemm=True),
        }
    )

    def turbo_provider(*args, **kwargs):
        provider = spec_provider.PrimusTurboSpecProvider(*args, **kwargs)
        provider.cfg = cfg
        return provider

    gpt_layer_specs.TESpecProvider = turbo_provider


def _gpt_moe(turbo: bool, seed: int, expert_parallel: int = 1):
    from megatron.core.models.gpt import GPTModel, gpt_layer_specs
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    te_provider = gpt_layer_specs.TESpecProvider
    _set_experts(turbo, te_provider)
    try:
        spec = gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec(
            num_experts=NUM_EXPERTS, moe_grouped_gemm=True
        )
    finally:
        gpt_layer_specs.TESpecProvider = te_provider
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=NUM_EXPERTS,
        moe_ffn_hidden_size=128,
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        expert_model_parallel_size=expert_parallel,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
    )
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    model = GPTModel(config=config, transformer_layer_spec=spec, vocab_size=128, max_sequence_length=64)
    return model.cuda()


def _expert_type(model):
    return type(model.decoder.layers[0].mlp.experts.linear_fc1).__name__


@pytest.fixture
def turbo_global_args(init_parallel_state):
    from megatron.training.global_vars import set_args

    set_args(SimpleNamespace(**GLOBAL_ARGS))


def test_turbo_experts_save_through_a_factory(turbo_global_args):
    """The trained ``weights`` (which the distributed optimizer looks up) replaces its
    per-expert ``weight{i}`` views in the checkpoint layout."""
    from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory

    model = _gpt_moe(turbo=True, seed=0)
    assert _expert_type(model) == "PrimusTurboGroupedLinear"
    sharded = model.sharded_state_dict(metadata={"singleton_local_shards": False})
    stacked = [k for k in model.state_dict() if k.endswith(".weights")]
    assert len(stacked) == 4  # fc1 and fc2 of both layers
    for key in stacked:
        assert isinstance(sharded[key], ShardedTensorFactory)
        assert sharded[key].data is model.get_parameter(key)
        assert not any(re.fullmatch(rf"{re.escape(key[:-1])}\d+", k) for k in sharded)

    with pytest.raises(AssertionError, match="singleton_local_shards=False"):
        model.sharded_state_dict(metadata={"singleton_local_shards": True})


@pytest.mark.parametrize("save_turbo", [False, True], ids=["te_to_turbo", "turbo_to_te"])
def test_torch_dist_moves_between_te_and_turbo(save_turbo, turbo_global_args, tmp_path):
    from megatron.core import dist_checkpointing

    source = _gpt_moe(turbo=save_turbo, seed=0)
    target = _gpt_moe(turbo=not save_turbo, seed=1)
    assert _expert_type(source) != _expert_type(target)
    metadata = {"singleton_local_shards": False}
    dist_checkpointing.save(
        {"model": source.sharded_state_dict(metadata=metadata)}, str(tmp_path), content_metadata=metadata
    )
    loaded = dist_checkpointing.load({"model": target.sharded_state_dict(metadata=metadata)}, str(tmp_path))
    target.load_state_dict(loaded["model"], strict=True)

    expected = {k: v for k, v in source.state_dict().items() if not k.endswith(("_extra_state", ".weights"))}
    state = target.state_dict()
    assert any(re.search(r"experts\.linear_fc1\.weight\d+$", k) for k in expected)
    for key, tensor in expected.items():
        torch.testing.assert_close(state[key], tensor, rtol=0, atol=0, msg=key)


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _model_and_optimizer(turbo, seed, expert_parallel):
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    gpt = _gpt_moe(turbo, seed, expert_parallel)
    model = DistributedDataParallel(
        gpt.config, DistributedDataParallelConfig(use_distributed_optimizer=True), gpt
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=1e-3,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
        ),
        [model],
    )
    return model, optimizer


def _save(model, optimizer, ckpt_dir):
    from megatron.core import dist_checkpointing

    model_sharded = model.sharded_state_dict()
    state = {
        "model": model_sharded,
        "optimizer": optimizer.sharded_state_dict(model_sharded, metadata=FULLY_RESHARDABLE),
    }
    dist_checkpointing.save(state, ckpt_dir)


def _reshard_worker(rank, port, save_turbo, tmp):
    """Save at EP 2 after one Adam step, load at EP 1 with the other expert implementation, save
    again, and compare the two checkpoints' plain tensors (weights, fp32 master weights, moments)."""
    import os

    import torch.distributed as dist
    from megatron.core import dist_checkpointing
    from megatron.core import parallel_state as ps
    from megatron.core.dist_checkpointing.dict_utils import diff
    from megatron.core.dist_checkpointing.serialization import load_plain_tensors
    from megatron.training.global_vars import set_args

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    set_args(SimpleNamespace(**GLOBAL_ARGS))
    first, second = os.path.join(tmp, "ep2"), os.path.join(tmp, "ep1")

    ps.initialize_model_parallel(expert_model_parallel_size=2)
    model, optimizer = _model_and_optimizer(save_turbo, seed=0, expert_parallel=2)
    ids = torch.randint(0, 128, (2, 64), generator=torch.Generator().manual_seed(rank)).cuda()
    positions = torch.arange(64, device="cuda").unsqueeze(0).expand(2, -1)
    model.zero_grad_buffer()
    optimizer.zero_grad()
    model(ids, positions, None, labels=torch.roll(ids, -1, dims=1)).float().mean().backward()
    model.finish_grad_sync()
    assert optimizer.step()[0], "Adam step failed"
    _save(model, optimizer, first)
    ps.destroy_model_parallel()

    ps.initialize_model_parallel(expert_model_parallel_size=1)
    model, optimizer = _model_and_optimizer(not save_turbo, seed=1, expert_parallel=1)
    model_sharded = model.sharded_state_dict()
    loaded = dist_checkpointing.load(
        {
            "model": model_sharded,
            "optimizer": optimizer.sharded_state_dict(
                model_sharded, is_loading=True, metadata=FULLY_RESHARDABLE
            ),
        },
        first,
    )
    model.load_state_dict(loaded["model"], strict=True)
    optimizer.load_state_dict(loaded["optimizer"])
    _save(model, optimizer, second)
    ps.destroy_model_parallel()

    if rank == 0:
        before, after = load_plain_tensors(first), load_plain_tensors(second)
        assert any("exp_avg" in k and "experts" in k for k in before), sorted(before)
        differences = diff(before, after)
        assert not any(map(bool, differences)), differences
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("save_turbo", [True, False], ids=["turbo_ep2_to_te_ep1", "te_ep2_to_turbo_ep1"])
def test_optimizer_state_reshards_across_expert_parallel(save_turbo, tmp_path):
    """Fully reshardable distributed-optimizer state, saved at EP 2, loads at EP 1 with the other
    expert implementation; this needs Turbo's ``weights`` in the checkpoint layout."""
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    mp.spawn(_reshard_worker, args=(_free_port(), save_turbo, str(tmp_path)), nprocs=2, join=True)
