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
    _get_decoupled_grads_for_grad_norm,
    _get_grad_norm_tensor,
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
        "primus.backends.megatron.patches.te_patches.fused_adam_clip_patches."
        "_get_grad_norm_tensor",
        lambda optimizer: norm,
    )
    optimizer = FakeChainedOptimizer()

    success, returned_norm, num_zeros = _chained_step_with_device_clip(
        optimizer, FakeAdam
    )

    assert success
    assert returned_norm is None
    assert num_zeros == 7.0
    assert optimizer.stepped
    assert [
        child.optimizer._primus_clip_norm for child in optimizer.chained_optimizers
    ] == [norm, norm]
    assert [
        child.optimizer._primus_clip_max_norm for child in optimizer.chained_optimizers
    ] == [1.0, 2.0]


def test_metadata_carries_bf16_writeback_address():
    grad = torch.zeros(5, dtype=torch.float32)
    param = torch.zeros(5, dtype=torch.float32)
    exp_avg = torch.zeros(5, dtype=torch.bfloat16)
    exp_avg_sq = torch.zeros(5, dtype=torch.bfloat16)
    model_param = torch.zeros(5, dtype=torch.bfloat16)

    addresses, sizes, block_map, chunk_offsets, chunks, moment_dtype, grad_dtype = (
        _build_metadata(
            4,
            [[grad], [param], [exp_avg], [exp_avg_sq]],
            {param.data_ptr(): model_param},
        )
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

    *_, moment_dtype, grad_dtype = _build_metadata(
        8, [[grad], [param], [exp_avg], [exp_avg_sq]]
    )

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
            "_get_model_param_range_map": lambda self, param: {
                "gbuf_world_in_bucket": Range()
            },
        },
    )()
    monkeypatch.setenv("PRIMUS_FUSED_ADAM_BF16_WRITEBACK", "1")

    _prepare_bf16_writeback(optimizer, FakeAdam)

    destination = optimizer.optimizer._primus_bf16_writeback_by_param_ptr[
        main_param.data_ptr()
    ]
    assert destination.data_ptr() == param_buffer[2:6].data_ptr()
    assert optimizer._primus_bf16_writeback_active
