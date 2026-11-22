# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import Mock, call

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


@pytest.fixture
def installed_pipeline(monkeypatch):
    # Only the optional package may be absent. A broken Megatron pin must fail.
    if find_spec("megatron") is None:
        pytest.skip("Megatron-LM is not installed")
    from megatron.core import recompute
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
    native_schedule = schedules.forward_backward_pipelining_without_interleaving
    native_decoder = transformer_block.checkpointed_forward
    entry.handler(ctx)
    assert transformer_block.checkpointed_forward is recompute.checkpointed_forward
    assert transformer_block.checkpointed_forward is not native_decoder
    sched = schedules.forward_backward_pipelining_without_interleaving
    assert sched._primus_pipeline_prefetch
    entry.handler(ctx)
    assert schedules.forward_backward_pipelining_without_interleaving is sched
    return schedules, transformer_block, native_schedule, native_decoder


@pytest.mark.parametrize("stage", [0, 1], ids=["first_stage", "last_stage"])
def test_pipeline_hooks_preserve_the_live_graph_and_random_sequence(installed_pipeline, monkeypatch, stage):
    from megatron.core import tensor_parallel, utils
    from megatron.core.transformer import cuda_graphs

    schedules, transformer_block, native_schedule, native_decoder = installed_pipeline

    class Layer(torch.nn.Linear):
        def forward(self, hidden_states, **kwargs):
            assert self.current_microbatch == current_mb
            assert timer.active
            visited.append((current_mb, self.layer_number))
            return super().forward(hidden_states).tanh() * torch.rand_like(hidden_states)

    layers = torch.nn.ModuleList([Layer(4, 4) for _ in range(6)])
    for i, layer in enumerate(layers):
        layer.layer_number = i + 1
    config = settings(
        overlap_p2p_comm=False,
        no_sync_func=None,
        grad_sync_func=None,
        finalize_model_grads_func=None,
        num_microbatches_with_partial_activation_checkpoints=None,
        deallocate_pipeline_outputs=False,
        barrier_with_L1_time=False,
        fp4=False,
    )
    model = SimpleNamespace(
        decoder=SimpleNamespace(layers=layers, config=config, num_layers_per_pipeline_rank=len(layers)),
        dp_cp_group=0,
        ep_group=1,
        expt_dp_group=2,
    )
    groups = Mock(spec=schedules.ProcessGroupCollection)
    groups.tp, groups.cp = None, Mock()
    p2p = Mock(total_stages=2, current_stage=stage, is_pp_first_stage=stage == 0, is_pp_last_stage=stage == 1)
    p2p.send_backward_recv_forward.return_value = None
    monkeypatch.setattr(schedules, "get_model_config", lambda _: config)
    monkeypatch.setattr(schedules, "get_tensor_shapes", lambda **kwargs: [])
    # The native schedule uses this scalar only to accumulate the token count.
    monkeypatch.setattr(schedules, "torch", SimpleNamespace(int=torch.int, zeros=lambda *a, **kw: 0))
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

    current_mb, visited, timer = None, [], SimpleNamespace(active=False)

    def run(schedule, decoder_forward, patched):
        nonlocal current_mb
        layers.zero_grad()
        torch.manual_seed(7)
        visited.clear()
        outputs, backward_positions, comm_events = [], [], []

        def set_timing(active):
            assert timer.active != active
            timer.active = active

        timer.start = lambda: set_timing(True)
        timer.stop = lambda: set_timing(False)
        config.timers = Mock(
            side_effect=lambda name, **kwargs: timer if name == "forward-compute" else Mock()
        )

        def forward(*args, current_microbatch, **kwargs):
            nonlocal current_mb
            current_mb = current_microbatch
            timer.start()
            set_mb(model, current_mb)
            state = decoder_forward(
                model.decoder, torch.rand(8, 4), None, None, None, None, None, None, False
            )
            outputs.append(state)
            timer.stop()
            return state, 0

        def backward(_input, output, _grad, _config):
            if stage == 0 and not backward_positions:
                assert all(layer.current_microbatch == 1 for layer in layers)
            assert not timer.active
            backward_positions.append(len(visited))
            output.sum().backward()

        monkeypatch.setattr(schedules, "forward_step", forward)
        monkeypatch.setattr(schedules, "backward_step", backward)
        monkeypatch.setattr(
            schedules, "_primus_prefetch_expert_parameters", lambda _: comm_events.append("prefetch")
        )
        p2p.recv_forward.side_effect = lambda *args: comm_events.append("recv")
        schedule(
            forward_step_func=None,
            data_iterator=None,
            model=model,
            num_microbatches=3,
            seq_length=1,
            micro_batch_size=1,
            p2p_communicator=p2p,
            pg_collection=groups,
        )
        if patched and stage == 1:
            assert comm_events == ["prefetch", "recv"]
        else:
            assert "prefetch" not in comm_events
        # Stage 0 normally runs B0 after F1. The patch moves it four layers into F2.
        if stage == 0:
            assert backward_positions == [
                2 * len(layers) + (4 if patched else 0),
                3 * len(layers),
                3 * len(layers),
            ]
            if patched:
                assert config.timers.call_args_list.count(call("forward-compute", log_level=2)) == 1
        else:
            assert backward_positions == [len(layers), 2 * len(layers), 3 * len(layers)]
        values = [*outputs, *(p.grad.clone() for p in layers.parameters()), torch.get_rng_state()]
        return values, visited.copy()

    reference, order = run(native_schedule, native_decoder, patched=False)
    actual, visits = run(
        schedules.forward_backward_pipelining_without_interleaving,
        transformer_block.checkpointed_forward,
        patched=True,
    )
    assert visits == order
    for observed, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
