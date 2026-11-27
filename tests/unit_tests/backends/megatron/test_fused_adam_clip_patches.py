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
    _deferred_passthrough_reduce_max,
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


def test_metadata_param_remainder_layout():
    grad = torch.zeros(9, dtype=torch.bfloat16)
    param = torch.zeros(9, dtype=torch.bfloat16)
    exp_avg = torch.zeros(9, dtype=torch.bfloat16)
    exp_avg_sq = torch.zeros(9, dtype=torch.bfloat16)
    remainder = torch.zeros(9, dtype=torch.int16)

    addresses, sizes, block_map, chunk_offsets, chunks, moment_dtype, grad_dtype = _build_metadata(
        8, [[grad], [param], [exp_avg], [exp_avg_sq], [remainder]], param_remainder=True
    )

    assert addresses.tolist() == [t.data_ptr() for t in (grad, param, exp_avg, exp_avg_sq, remainder)]
    assert sizes.tolist() == [9]
    assert block_map.tolist() == [0, 0]
    assert chunk_offsets.tolist() == [0]
    assert chunks == 2
    assert (moment_dtype, grad_dtype) == (1, 1)


def test_metadata_param_remainder_rejects_fp32_params():
    tensors = [
        [torch.zeros(8, dtype=dt)] for dt in (torch.bfloat16, torch.float32, torch.bfloat16, torch.bfloat16)
    ]
    with pytest.raises(RuntimeError, match="int16 remainders"):
        _build_metadata(8, tensors + [[torch.zeros(8, dtype=torch.int16)]], param_remainder=True)
    with pytest.raises(RuntimeError, match="five-list"):
        _build_metadata(8, tensors, param_remainder=True)


_ADAM = dict(lr=3e-4, beta1=0.9, beta2=0.95, eps=1e-8, step=7, mode=1, bias_correction=1, weight_decay=0.1)


def _remainder_state(sizes, seed):
    """BF16 grads / params / moments and int16 remainders, as TE keeps them, for tensors of ``sizes``."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    lists = [[], [], [], [], []]
    for n in sizes:
        master = (0.02 * torch.randn(n, generator=g)).cuda()
        bits = master.view(torch.int32)
        lo = (bits & 0xFFFF).to(torch.int16)
        hi = ((bits >> 16) + (lo < 0).to(torch.int32)).to(torch.int16)
        lists[0].append((1e-3 * torch.randn(n, generator=g)).to("cuda", torch.bfloat16))
        lists[1].append(hi.view(torch.bfloat16))
        lists[2].append((1e-4 * torch.randn(n, generator=g)).to("cuda", torch.bfloat16))
        lists[3].append((1e-8 * torch.rand(n, generator=g)).to("cuda", torch.bfloat16))
        lists[4].append(lo)
    return lists


def _te_remainder_step(lists):
    tex = pytest.importorskip("transformer_engine_torch")
    from transformer_engine.pytorch.optimizers import multi_tensor_applier

    noop = torch.zeros(1, dtype=torch.int, device="cuda")
    a = _ADAM
    multi_tensor_applier(
        tex.multi_tensor_adam_param_remainder, noop, lists, a["lr"], a["beta1"], a["beta2"], a["eps"], a["step"],
        a["mode"], a["bias_correction"], a["weight_decay"],
    )  # fmt: skip


def _fused_remainder_step(lists, grad_norm, max_norm=1.0, chunk_size=2048 * 32):
    from primus.backends.megatron.patches.te_patches.fused_adam_clip_patches import (
        _load_extension,
    )

    addresses, sizes, block_map, offsets, chunks, moment_dtype, grad_dtype = _build_metadata(
        chunk_size, lists, param_remainder=True
    )
    a = _ADAM
    _load_extension().fused_adam_clip(
        addresses, sizes, block_map, offsets, chunks, chunk_size, a["lr"], a["beta1"], a["beta2"], a["eps"],
        a["step"], a["mode"], a["bias_correction"], a["weight_decay"], moment_dtype, grad_dtype,
        torch.tensor([grad_norm], dtype=torch.float32, device="cuda"), max_norm, 1,
    )  # fmt: skip


def _clone(lists):
    return [[t.clone() for t in tensors] for tensors in lists]


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("needs a ROCm / CUDA device")


def test_param_remainder_kernel_matches_te_without_clipping():
    """A norm below max_norm: coefficient 1, so every output must equal TE's ``multi_tensor_adam_param_remainder``
    bit for bit; sizes cover the 8-wide vector path, its tail and a tensor over one chunk."""
    _require_gpu()
    ref = _remainder_state([8, 13, 4096 + 5, 2048 * 32 + 24], seed=0)
    got = _clone(ref)
    _te_remainder_step(ref)
    _fused_remainder_step(got, grad_norm=0.5)
    for name, r, g in zip(("grad", "param", "exp_avg", "exp_avg_sq", "remainder"), ref, got):
        for i, (a, b) in enumerate(zip(r, g)):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"{name}[{i}] differs"


def test_param_remainder_kernel_clips_like_te_on_scaled_fp32_grads():
    """A norm above max_norm: the kernel scales each BF16 gradient by max_norm / (norm + 1e-6) in FP32 as it loads
    it, which is TE's step on those FP32-scaled gradients."""
    _require_gpu()
    norm, max_norm = 4.0, 1.0
    base = _remainder_state([4096 + 3, 2048 * 32 + 8], seed=1)
    got = _clone(base)
    _fused_remainder_step(got, grad_norm=norm, max_norm=max_norm)
    coeff = torch.tensor(max_norm, dtype=torch.float32) / (torch.tensor(norm, dtype=torch.float32) + 1e-6)
    ref = _clone(base)
    ref[0] = [g.float() * coeff.cuda() for g in ref[0]]
    try:
        _te_remainder_step(ref)
    except RuntimeError as exc:  # a TE build whose remainder op takes BF16 gradients only
        pytest.skip(f"TE's param-remainder Adam does not take FP32 gradients here: {exc}")
    for name, r, g in zip(("param", "exp_avg", "exp_avg_sq", "remainder"), ref[1:], got[1:]):
        for i, (a, b) in enumerate(zip(r, g)):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"{name}[{i}] differs"
    # and the clip changed something: without it the update is TE's unscaled one
    plain = _clone(base)
    _te_remainder_step(plain)
    assert not all(torch.equal(a, b) for a, b in zip(plain[1], got[1]))


def test_param_remainder_kernel_matches_te_on_misaligned_shards():
    """Distributed-optimizer shards are views at any element offset of a flat buffer: views 1 and 3 elements in
    (2 and 6 bytes past a 16-byte boundary) must still match TE bit for bit."""
    _require_gpu()
    sizes = [4096 + 7, 2048 * 32 + 3]
    full = _remainder_state([n + 3 for n in sizes], seed=2)
    ref = _clone(full)
    got = _clone(full)

    def views(lists, offset):
        return [[t[offset : offset + n] for t, n in zip(tensors, sizes)] for tensors in lists]

    for offset in (1, 3):
        ref_v, got_v = views(ref, offset), views(got, offset)
        assert all(t.data_ptr() % 16 for tensors in got_v for t in tensors)
        _te_remainder_step(ref_v)
        _fused_remainder_step(got_v, grad_norm=0.5)
    for name, r, g in zip(("grad", "param", "exp_avg", "exp_avg_sq", "remainder"), ref, got):
        for i, (a, b) in enumerate(zip(r, g)):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"{name}[{i}] differs"
