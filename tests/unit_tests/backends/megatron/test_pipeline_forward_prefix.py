# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from primus.backends.megatron.core.pipeline_parallel import forward_prefix as prefix
from primus.backends.megatron.patches.parallelism.forward_prefix_patches import (
    validate_pipeline_prefetch,
)


def settings(**overrides):
    values = dict(
        pipeline_forward_prefix_layers=4,
        early_expert_param_gather=True,
        pipeline_model_parallel_size=2,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        decoder_first_pipeline_num_layers=20,
        num_experts=128,
        enable_primus_turbo=True,
        use_distributed_optimizer=True,
        overlap_param_gather=True,
        align_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        num_distributed_optimizer_instances=1,
        fp8="hybrid",
        fp8_recipe="tensorwise",
        recompute_granularity="full",
        recompute_method="block",
        recompute_num_layers=0,
        hidden_dropout=0,
        attention_dropout=0,
    )
    return SimpleNamespace(**(values | overrides))


def test_disabled_features_have_no_model_requirements():
    validate_pipeline_prefetch(SimpleNamespace())


@pytest.mark.parametrize("even", [False, True])
def test_router_and_data_are_independent(even):
    validate_pipeline_prefetch(settings(moe_router_force_load_balancing=even, mock_data=False))


@pytest.mark.parametrize(
    "changes",
    [
        {"pipeline_forward_prefix_layers": -1},
        {"pipeline_forward_prefix_layers": 20},
        {"fp8_recipe": "delayed"},
        {"attention_dropout": 0.1},
        {"recompute_num_layers": 1},
        {"recompute_layer_ids": []},
        {"enable_primus_turbo": False},
        {"num_experts": None},
        {"num_microbatches_with_partial_activation_checkpoints": 0},
        {"enable_cuda_graph": True},
        {"external_cuda_graph": True},
        {"cuda_graph_impl": "local"},
        {"cuda_graph_impl": "transformer_engine"},
        {"cuda_graph_impl": "full_iteration"},
        {"num_layers_per_virtual_pipeline_stage": 1},
        {"context_parallel_size": 2},
        {"align_param_gather": True},
        {"overlap_param_gather_with_optimizer_step": True},
    ],
)
def test_incompatible_schedule_is_rejected(changes):
    with pytest.raises(ValueError):
        validate_pipeline_prefetch(settings(**changes))


def test_prefetch_preserves_native_bucket_ownership():
    model = Mock()
    group = SimpleNamespace(
        next_param_gather_bucket_group=None,
        param_gather_dispatched=False,
        intra_distributed_optimizer_instance_group=model.intra_expt_dp_group,
    )
    model.expert_parallel_bucket_groups = [group]
    model.remove_forward_pre_hook_handles = {}
    prefix.prefetch_expert_parameters(model)
    model._start_bucket_group_param_sync.assert_not_called()
    model.remove_forward_pre_hook_handles = {"weight": object()}
    prefix.prefetch_expert_parameters(model)
    group.param_gather_dispatched = True
    prefix.prefetch_expert_parameters(model)
    model._start_bucket_group_param_sync.assert_called_once_with(group, force_sync=False)
    model.expert_parallel_bucket_groups = [group, group]
    with pytest.raises(ValueError, match="unchained"):
        prefix.prefetch_expert_parameters(model)


def test_install_binds_the_real_schedule_and_decoder_alias(monkeypatch):
    recompute = pytest.importorskip("megatron.core.recompute")
    from megatron.core.pipeline_parallel import schedules
    from megatron.core.transformer import transformer_block

    from primus.core.patches import PatchContext, PatchRegistry

    params = settings()
    ctx = PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=params)}
    )
    entry = PatchRegistry.get("megatron.pp.pipeline_prefetch")
    assert entry is not None and entry.condition(ctx)
    for module, name in (
        (schedules, "forward_backward_pipelining_without_interleaving"),
        (recompute, "checkpointed_forward"),
        (transformer_block, "checkpointed_forward"),
    ):
        monkeypatch.setattr(module, name, getattr(module, name))
    for module, names in (
        (schedules, ["_primus_forward_prefix", "_primus_partial", "_primus_prefetch_expert_parameters"]),
        (recompute, ["_primus_after_decoder_layer"]),
    ):
        for name in names:
            monkeypatch.setattr(module, name, None, raising=False)
    original = transformer_block.checkpointed_forward
    entry.handler(ctx)
    assert transformer_block.checkpointed_forward is recompute.checkpointed_forward
    assert transformer_block.checkpointed_forward is not original
    sched = schedules.forward_backward_pipelining_without_interleaving
    assert sched._primus_pipeline_prefetch
    entry.handler(ctx)
    assert schedules.forward_backward_pipelining_without_interleaving is sched


def test_cut_keeps_the_live_graph_microbatch_and_random_sequence(monkeypatch):
    from megatron.core import tensor_parallel, utils
    from megatron.core.transformer import cuda_graphs

    layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(6)])
    model = SimpleNamespace(
        decoder=SimpleNamespace(layers=layers), dp_cp_group=0, ep_group=1, expt_dp_group=2
    )
    monkeypatch.setattr(utils, "get_attr_wrapped_model", lambda m, k, **kw: getattr(m, k))
    tracker = Mock()
    tracker.get_states.return_value = {}
    monkeypatch.setattr(tensor_parallel, "get_cuda_rng_tracker", lambda: tracker)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0])
    gpu_rng = torch.Generator().manual_seed(9)
    monkeypatch.setattr(torch.cuda, "get_rng_state", gpu_rng.get_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", gpu_rng.set_state)
    monkeypatch.setattr(prefix, "_ambient_state", lambda: (True, False, False, None, 0, (False,) * 9))

    def set_mb(model, ordinal):
        for layer in model.decoder.layers:
            layer.current_microbatch = ordinal

    monkeypatch.setattr(cuda_graphs, "set_current_microbatch", set_mb)

    def run(depth):
        layers.zero_grad()
        torch.manual_seed(7)
        visited = []
        timer = SimpleNamespace(active=False)

        def set_timing(active):
            assert timer.active != active
            timer.active = active

        timer.start = lambda: set_timing(True)
        timer.stop = lambda: set_timing(False)
        timers = Mock(return_value=timer)

        def forward(ordinal):
            timer.start()
            set_mb(model, ordinal)
            state = torch.rand(8, 4)
            for i, layer in enumerate(layers):
                assert layer.current_microbatch == ordinal
                assert timer.active
                state = layer(state).tanh() * torch.rand_like(state)
                visited.append((ordinal, i))
                prefix.after_decoder_layer(model.decoder, i + 1)
            timer.stop()
            return state

        f0 = forward(0)
        f1 = forward(1)

        def backward0():
            assert all(layer.current_microbatch == 1 for layer in layers)
            assert not timer.active
            f0.sum().backward()

        if depth:
            gate = prefix.ForwardPrefix(model, depth, forward, timers)
            gate.defer(backward0)
            f2 = gate.forward(2)
            assert gate.handoff is None
            timers.assert_called_once_with("forward-compute", log_level=2)
        else:
            backward0()
            f2 = forward(2)
        f1.sum().backward()
        f2.sum().backward()
        return [f0, f1, f2, *(p.grad.clone() for p in layers.parameters()), torch.get_rng_state()], visited

    reference, order = run(0)
    actual, visits = run(4)
    assert visits == order
    for observed, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
