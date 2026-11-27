###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus.backends.megatron.patches.te_patches.fused_adam_clip_patches import (
    _build_metadata,
    _chained_step_with_device_clip,
    _clip_coeff,
    _copy_model_grads_to_decoupled,
    _defer_expert_grad_scaling,
    _deferred_passthrough_reduce_max,
    _get_decoupled_grads_for_grad_norm,
    _get_grad_norm_tensor,
    _grad_prescale,
    _load_extension,
    _prepare_bf16_writeback,
    _uses_static_bf16_unity_scale,
)


@pytest.mark.parametrize(
    ("total_norm", "max_norm", "expected"),
    [
        (0.25, 1.0, 1.0),
        (1.0, 1.0, 1.0 / 1.000001),
        (2.0, 1.0, 1.0 / 2.000001),
        (8.0, 2.0, 2.0 / 8.000001),
    ],
)
def test_clip_coeff(total_norm, max_norm, expected):
    assert _clip_coeff(total_norm, max_norm) == pytest.approx(expected)


def test_grad_norm_stays_on_device(monkeypatch):
    from megatron.core.optimizer import clip_grads

    class FakeOptimizer:
        def __init__(self):
            self.grad = torch.ones(4, dtype=torch.float32)
            self.group = object()

        def get_main_grads_for_grad_norm(self):
            return [self.grad]

        def get_grad_stats_parallel_group(self):
            return self.group

    optimizer = FakeOptimizer()
    monkeypatch.setattr(
        clip_grads,
        "multi_tensor_applier",
        lambda impl, overflow, tensor_lists, per_parameter: (
            torch.tensor([3.0]),
            None,
        ),
    )

    def fake_all_reduce(value, op, group):
        assert op == torch.distributed.ReduceOp.SUM
        assert group is optimizer.group
        value.mul_(4.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    norm = _get_grad_norm_tensor(optimizer)

    assert isinstance(norm, torch.Tensor)
    assert norm.device == optimizer.grad.device
    assert torch.equal(norm, torch.tensor([6.0]))
    assert optimizer._primus_norm_overflow_buf.dtype == torch.int32


def test_chained_step_forwards_device_norm_to_each_adam(monkeypatch):
    class FakeAdam:
        pass

    class FakeChildOptimizer:
        def __init__(self, clip_grad):
            self.optimizer = FakeAdam()
            self.config = type("Config", (), {"clip_grad": clip_grad})()

        def get_parameters(self):
            return [object()]

    class FakeChainedOptimizer:
        def __init__(self):
            self.chained_optimizers = [FakeChildOptimizer(1.0), FakeChildOptimizer(2.0)]
            self.config = type("Config", (), {"log_num_zeros_in_grad": True})()
            self.stepped = False

        def prepare_grads(self):
            return False

        def grads_states_parallel_group_is_shared(self):
            return True

        def count_zeros(self):
            return 7.0

        def step_with_ready_grads(self):
            self.stepped = True
            return True

    norm = torch.tensor([3.0])
    monkeypatch.setattr(
        "primus.backends.megatron.patches.te_patches.fused_adam_clip_patches." "_get_grad_norm_tensor",
        lambda optimizer: norm,
    )
    optimizer = FakeChainedOptimizer()

    success, returned_norm, num_zeros = _chained_step_with_device_clip(optimizer, FakeAdam)

    assert success
    assert returned_norm is None
    assert num_zeros == 7.0
    assert optimizer.stepped
    assert [child.optimizer._primus_clip_norm for child in optimizer.chained_optimizers] == [norm, norm]
    assert [child.optimizer._primus_clip_max_norm for child in optimizer.chained_optimizers] == [1.0, 2.0]


def test_grad_norm_uses_megatron_019_getter(monkeypatch):
    from megatron.core.optimizer import clip_grads

    class FakeOptimizer:
        def __init__(self):
            self.grad = torch.ones(4, dtype=torch.float32)

        def get_grads_for_grad_norm(self):
            return [self.grad]

        def get_grad_stats_parallel_group(self):
            return None

    seen = []
    monkeypatch.setattr(
        clip_grads,
        "multi_tensor_applier",
        lambda impl, overflow, tensor_lists, per_parameter: (
            seen.append(tensor_lists) or torch.tensor([3.0]),
            None,
        ),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value, op, group: None)

    optimizer = FakeOptimizer()
    assert torch.equal(_get_grad_norm_tensor(optimizer), torch.tensor([3.0]))
    assert seen == [[[optimizer.grad]]]


def test_deferred_reduce_max_takes_megatron_019_group():
    norm = torch.tensor([3.0])

    assert _deferred_passthrough_reduce_max(norm) is norm
    assert _deferred_passthrough_reduce_max(norm, group=object()) is norm
    assert _deferred_passthrough_reduce_max(None, group=None) is None


def _chained_optimizer(skip_threshold=float("inf"), grouped=False):
    class FakeAdam:
        pass

    class FakeChildOptimizer:
        def __init__(self):
            self.optimizer = FakeAdam()
            self.config = type("Config", (), {"clip_grad": 1.0, "grad_norm_skip_threshold": skip_threshold})()

        def get_parameters(self):
            return [object()]

    class FakeChainedOptimizer:
        def __init__(self):
            self.chained_optimizers = [FakeChildOptimizer()]
            self.config = type("Config", (), {"log_num_zeros_in_grad": False})()

        def prepare_grads(self):
            return False

        def grads_states_parallel_group_is_shared(self):
            return True

        def has_grad_norm_group(self, group):
            return grouped

        def step_with_ready_grads(self):
            return True

    return FakeChainedOptimizer(), FakeAdam


def test_chained_step_rejects_grad_norm_skip_threshold(monkeypatch):
    monkeypatch.setattr(
        "primus.backends.megatron.patches.te_patches.fused_adam_clip_patches._get_grad_norm_tensor",
        lambda optimizer: torch.tensor([3.0]),
    )
    optimizer, adam_type = _chained_optimizer(skip_threshold=10.0)

    with pytest.raises(RuntimeError, match="grad_norm_skip_threshold"):
        _chained_step_with_device_clip(optimizer, adam_type)


def test_chained_step_rejects_separate_grad_norm_group(monkeypatch):
    monkeypatch.setattr(
        "primus.backends.megatron.patches.te_patches.fused_adam_clip_patches._separate_grad_norm_groups",
        lambda: ("mtp",),
    )
    optimizer, adam_type = _chained_optimizer(grouped=True)

    with pytest.raises(RuntimeError, match="separate grad-norm group"):
        _chained_step_with_device_clip(optimizer, adam_type)


def test_metadata_carries_bf16_writeback_address():
    grad = torch.zeros(5, dtype=torch.float32)
    param = torch.zeros(5, dtype=torch.float32)
    exp_avg = torch.zeros(5, dtype=torch.bfloat16)
    exp_avg_sq = torch.zeros(5, dtype=torch.bfloat16)
    model_param = torch.zeros(5, dtype=torch.bfloat16)

    addresses, sizes, block_map, chunk_offsets, chunks, moment_dtype, grad_dtype = _build_metadata(
        4,
        [[grad], [param], [exp_avg], [exp_avg_sq]],
        {param.data_ptr(): model_param},
    )

    assert addresses.tolist() == [
        grad.data_ptr(),
        param.data_ptr(),
        exp_avg.data_ptr(),
        exp_avg_sq.data_ptr(),
        model_param.data_ptr(),
    ]
    assert sizes.tolist() == [5]
    assert block_map.tolist() == [0, 0]
    assert chunk_offsets.tolist() == [0]
    assert chunks == 2
    assert moment_dtype == 1
    assert grad_dtype == 0


def test_metadata_accepts_direct_bf16_gradient():
    grad = torch.zeros(8, dtype=torch.bfloat16)
    param = torch.zeros(8, dtype=torch.float32)
    exp_avg = torch.zeros(8, dtype=torch.bfloat16)
    exp_avg_sq = torch.zeros(8, dtype=torch.bfloat16)

    *_, moment_dtype, grad_dtype = _build_metadata(8, [[grad], [param], [exp_avg], [exp_avg_sq]])

    assert moment_dtype == 1
    assert grad_dtype == 1


def test_copy_model_grads_attaches_bf16_view_without_cast():
    class Range:
        start = 2
        end = 6
        size = 4

    model_param = torch.zeros(4, dtype=torch.bfloat16)
    model_param.main_grad = torch.arange(8, dtype=torch.bfloat16)
    main_param = torch.zeros(4, dtype=torch.float32)
    optimizer = type(
        "FakeDistributedOptimizer",
        (),
        {
            "is_stub_optimizer": False,
            "ddp_config": type("DDPConfig", (), {"use_megatron_fsdp": False})(),
            "model_float16_groups": [[model_param]],
            "shard_fp32_from_float16_groups": [[main_param]],
            "model_fp32_groups": [],
            "shard_fp32_groups": [],
            "_get_model_param_range_map": lambda self, param: {"param": Range()},
        },
    )()

    _copy_model_grads_to_decoupled(optimizer)

    assert main_param.decoupled_grad.dtype == torch.bfloat16
    assert main_param.decoupled_grad.data_ptr() == model_param.main_grad[2:6].data_ptr()
    assert torch.equal(main_param.decoupled_grad, model_param.main_grad[2:6])


def test_decoupled_grad_norm_filter_uses_transformer_module_helper(monkeypatch):
    from megatron.core import tensor_parallel

    kept = torch.nn.Parameter(torch.zeros(2))
    kept.decoupled_grad = torch.ones(2, dtype=torch.bfloat16)
    shared = torch.nn.Parameter(torch.zeros(2))
    shared.decoupled_grad = torch.ones(2, dtype=torch.bfloat16)
    shared.shared = True
    monkeypatch.setattr(
        tensor_parallel,
        "param_is_not_tensor_parallel_duplicate",
        lambda param, group: True,
    )
    optimizer = type(
        "FakeOptimizer",
        (),
        {"get_parameters": lambda self: [kept, shared], "tp_group": object()},
    )()

    grads = _get_decoupled_grads_for_grad_norm(optimizer)

    assert grads == [kept.decoupled_grad]


def test_static_bf16_unity_scale_requires_bf16_without_grad_scaler():
    static = type(
        "StaticOptimizer",
        (),
        {"config": type("Config", (), {"bf16": True})(), "grad_scaler": None},
    )()
    scaled = type(
        "ScaledOptimizer",
        (),
        {"config": type("Config", (), {"bf16": True})(), "grad_scaler": object()},
    )()
    chained = type("Chained", (), {"chained_optimizers": [static]})()

    assert _uses_static_bf16_unity_scale(static)
    assert _uses_static_bf16_unity_scale(chained)
    assert not _uses_static_bf16_unity_scale(scaled)


def test_prepare_bf16_writeback_maps_master_to_param_buffer(monkeypatch):
    class FakeAdam:
        pass

    class Range:
        start = 2
        end = 6
        size = 4

    main_param = torch.zeros(4, dtype=torch.float32)
    model_param = torch.zeros(4, dtype=torch.bfloat16)
    param_buffer = torch.zeros(8, dtype=torch.bfloat16)
    bucket = type("Bucket", (), {"param_data": param_buffer})()
    buffer = type("Buffer", (), {"buckets": [bucket]})()
    optimizer = type(
        "FakeDistributedOptimizer",
        (),
        {
            "is_stub_optimizer": False,
            "ddp_config": type("DDPConfig", (), {"use_megatron_fsdp": False})(),
            "config": type(
                "Config",
                (),
                {"use_precision_aware_optimizer_no_fp8_or_ds_fp8": False},
            )(),
            "optimizer": FakeAdam(),
            "shard_fp32_from_float16_groups": [[main_param]],
            "model_float16_groups": [[model_param]],
            "model_param_gbuf_map": {model_param: (0, None, 0)},
            "buffers": [buffer],
            "_get_model_param_range_map": lambda self, param: {"gbuf_world_in_bucket": Range()},
        },
    )()
    monkeypatch.setenv("PRIMUS_FUSED_ADAM_BF16_WRITEBACK", "1")

    _prepare_bf16_writeback(optimizer, FakeAdam)

    destination = optimizer.optimizer._primus_bf16_writeback_by_param_ptr[main_param.data_ptr()]
    assert destination.data_ptr() == param_buffer[2:6].data_ptr()
    assert optimizer._primus_bf16_writeback_active


def _grad_buffer(factor, params):
    bucket = type("Bucket", (), {"gradient_scaling_factor": factor})()
    return type("Buffer", (), {"gradient_scaling_factor": factor, "buckets": [bucket], "params": params})()


def _model_optimizer(params):
    return type(
        "FakeDistributedOptimizer",
        (),
        {"model_float16_groups": [params], "model_fp32_groups": []},
    )()


def test_defer_expert_grad_scaling_moves_factor_to_params():
    expert = torch.nn.Parameter(torch.zeros(2))
    unscaled = torch.nn.Parameter(torch.zeros(2))
    scaled_buffer = _grad_buffer(0.25, [expert])
    unity_buffer = _grad_buffer(1.0, [unscaled])
    ddp = type("FakeDDP", (), {"expert_parallel_buffers": [scaled_buffer, unity_buffer]})()

    _defer_expert_grad_scaling(ddp)

    assert scaled_buffer.gradient_scaling_factor == 1.0
    assert scaled_buffer.buckets[0].gradient_scaling_factor == 1.0
    assert expert._primus_deferred_grad_scale == 0.25
    assert not hasattr(unscaled, "_primus_deferred_grad_scale")


def test_grad_prescale_reads_deferred_factor(monkeypatch):
    expert = torch.nn.Parameter(torch.zeros(2))
    expert._primus_deferred_grad_scale = 0.25
    dense = torch.nn.Parameter(torch.zeros(2))

    assert _grad_prescale(_model_optimizer([expert])) == 1.0
    monkeypatch.setenv("PRIMUS_FUSED_ADAM_EXPERT_GRAD_SCALE", "1")
    assert _grad_prescale(_model_optimizer([expert])) == 0.25
    assert _grad_prescale(_model_optimizer([dense])) == 1.0
    with pytest.raises(RuntimeError, match="several deferred gradient scales"):
        _grad_prescale(_model_optimizer([expert, dense]))


def test_grad_norm_weights_deferred_expert_scale(monkeypatch):
    from megatron.core.optimizer import clip_grads

    monkeypatch.setenv("PRIMUS_FUSED_ADAM_EXPERT_GRAD_SCALE", "1")
    expert = torch.nn.Parameter(torch.zeros(2))
    expert._primus_deferred_grad_scale = 0.25

    class FakeChild:
        def __init__(self, params, grad):
            self.model_float16_groups = [params]
            self.model_fp32_groups = []
            self.grad = grad

        def get_grads_for_grad_norm(self):
            return [self.grad]

    dense_child = FakeChild([torch.nn.Parameter(torch.zeros(2))], torch.tensor([3.0, 0.0]))
    expert_child = FakeChild([expert], torch.tensor([0.0, 16.0]))
    chained = type(
        "FakeChained",
        (),
        {
            "chained_optimizers": [dense_child, expert_child],
            "get_grad_stats_parallel_group": lambda self: None,
        },
    )()
    monkeypatch.setattr(
        clip_grads,
        "multi_tensor_applier",
        lambda impl, overflow, tensor_lists, per_parameter: (
            torch.linalg.vector_norm(torch.cat(tensor_lists[0])).reshape(1),
            None,
        ),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value, op, group: None)

    # sqrt(3^2 + (16 / 4)^2)
    assert torch.allclose(_get_grad_norm_tensor(chained), torch.tensor([5.0]))


def test_chained_step_forwards_deferred_scale_to_each_adam(monkeypatch):
    monkeypatch.setenv("PRIMUS_FUSED_ADAM_EXPERT_GRAD_SCALE", "1")
    monkeypatch.setattr(
        "primus.backends.megatron.patches.te_patches.fused_adam_clip_patches._get_grad_norm_tensor",
        lambda optimizer: torch.tensor([3.0]),
    )
    optimizer, adam_type = _chained_optimizer()
    child = optimizer.chained_optimizers[0]
    expert = torch.nn.Parameter(torch.zeros(2))
    expert._primus_deferred_grad_scale = 0.25
    child.model_float16_groups = [[expert]]
    child.model_fp32_groups = []

    _chained_step_with_device_clip(optimizer, adam_type)

    assert child.optimizer._primus_grad_prescale == 0.25


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_kernel_grad_prescale_matches_prescaled_grad():
    extension = _load_extension()
    torch.manual_seed(0)
    n = 4099
    grad = torch.randn(n, device="cuda")
    param = torch.randn(n, device="cuda")
    exp_avg = torch.randn(n, device="cuda").abs() * 1e-3
    exp_avg_sq = torch.randn(n, device="cuda").abs() * 1e-6
    norm = torch.tensor([2.0], device="cuda")

    def step(g, prescale):
        tensors = [g.clone(), param.clone(), exp_avg.clone(), exp_avg_sq.clone()]
        metadata = _build_metadata(1024, [[t] for t in tensors])
        extension.fused_adam_clip(
            *metadata[:5], 1024, 1e-3, 0.9, 0.95, 1e-8, 3, 1, 1, 0.1, *metadata[5:], norm, 1.0, prescale
        )
        return tensors[1:]

    for got, want in zip(step(grad, 0.25), step(grad * 0.25, 1.0)):
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-7)
