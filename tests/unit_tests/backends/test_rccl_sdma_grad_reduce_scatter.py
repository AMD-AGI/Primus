###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.core.distributed import rccl_sdma_param_gather
from primus.backends.megatron.patches.parallelism import (
    rccl_sdma_grad_reduce_scatter_patches as grad_patches,
)
from primus.backends.megatron.patches.parallelism import (
    rccl_sdma_param_all_gather_patches as param_patches,
)


def _bucket_group(**overrides):
    defaults = {
        "ddp_config": SimpleNamespace(
            use_distributed_optimizer=True,
            num_distributed_optimizer_instances=1,
            reduce_scatter_with_fp32_accumulation=False,
            check_for_nan_in_grad=False,
            check_for_large_grads=False,
            average_in_collective=False,
            overlap_grad_reduce=True,
        ),
        "is_first_batch": False,
        "grad_reduce_handle": None,
        "cached_grad_buffer_shard_list": [None],
        "intra_distributed_optimizer_instance_size": 2,
        "intra_distributed_optimizer_instance_rank": 0,
        "intra_distributed_optimizer_instance_group": SimpleNamespace(),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_start_grad_sync_uses_dedicated_group_when_eligible(monkeypatch):
    dedicated_group = SimpleNamespace()
    native_handle = SimpleNamespace()
    coalescing_calls = []
    rs_calls = []

    grad_data = torch.zeros(8)
    rccl_sdma_param_gather.mark_direct_param_buffer(grad_data)
    bucket = SimpleNamespace(grad_data=grad_data, gradient_scaling_factor=1.0)
    bg = _bucket_group(buckets=[bucket])

    def fake_coalescing_manager(group, async_ops):
        coalescing_calls.append((group, async_ops))
        return nullcontext(native_handle)

    def fake_reduce_scatter(output, input_, op, group, async_op):
        rs_calls.append((output, input_, op, group, async_op))

    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_coalescing_manager",
        fake_coalescing_manager,
    )
    monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", fake_reduce_scatter)
    monkeypatch.setattr(rccl_sdma_param_gather, "get_sdma_process_group", lambda _group: dedicated_group)

    wrapped = grad_patches.make_start_grad_sync(
        lambda *_a, **_k: pytest.fail("native Megatron fallback must not run")
    )
    wrapped(bg)

    assert coalescing_calls == [(dedicated_group, True)]
    assert len(rs_calls) == 1
    assert rs_calls[0][0] is bg.cached_grad_buffer_shard_list[0][bg.intra_distributed_optimizer_instance_rank]
    assert rs_calls[0][1] is grad_data
    assert rs_calls[0][2] == torch.distributed.ReduceOp.SUM
    assert rs_calls[0][3] is dedicated_group
    assert rs_calls[0][4] is True
    assert bg.grad_reduce_handle is native_handle


def test_start_grad_sync_average_in_collective_selects_avg(monkeypatch):
    rs_calls = []
    grad_data = torch.zeros(8)
    rccl_sdma_param_gather.mark_direct_param_buffer(grad_data)
    bucket = SimpleNamespace(grad_data=grad_data, gradient_scaling_factor=1.0)
    bg = _bucket_group(buckets=[bucket])
    bg.ddp_config.average_in_collective = True

    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_coalescing_manager",
        lambda group, async_ops: nullcontext(SimpleNamespace()),
    )
    monkeypatch.setattr(
        torch.distributed,
        "reduce_scatter_tensor",
        lambda output, input_, op, group, async_op: rs_calls.append(op),
    )
    monkeypatch.setattr(rccl_sdma_param_gather, "get_sdma_process_group", lambda _g: SimpleNamespace())

    wrapped = grad_patches.make_start_grad_sync(lambda *_a, **_k: pytest.fail("fallback must not run"))
    wrapped(bg)

    assert rs_calls == [torch.distributed.ReduceOp.AVG]


def test_start_grad_sync_applies_gradient_scaling_before_collective(monkeypatch):
    grad_data = torch.full((8,), 2.0)
    rccl_sdma_param_gather.mark_direct_param_buffer(grad_data)
    bucket = SimpleNamespace(grad_data=grad_data, gradient_scaling_factor=0.5)
    bg = _bucket_group(buckets=[bucket])

    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_coalescing_manager",
        lambda group, async_ops: nullcontext(SimpleNamespace()),
    )
    monkeypatch.setattr(
        torch.distributed,
        "reduce_scatter_tensor",
        lambda output, input_, op, group, async_op: None,
    )
    monkeypatch.setattr(rccl_sdma_param_gather, "get_sdma_process_group", lambda _g: SimpleNamespace())

    wrapped = grad_patches.make_start_grad_sync(lambda *_a, **_k: pytest.fail("fallback must not run"))
    wrapped(bg)

    assert torch.equal(grad_data, torch.full((8,), 1.0))


def test_start_grad_sync_waits_for_cudagraph_wgrad_events_before_collective(
    monkeypatch,
):
    order = []
    event = object()

    class ConsumerStream:
        def wait_event(self, waited_event):
            assert waited_event is event
            order.append("wait")

    grad_data = torch.zeros(8)
    rccl_sdma_param_gather.mark_direct_param_buffer(grad_data)
    param = SimpleNamespace(_cudagraph_wgrad_ready_event=event)
    bucket = SimpleNamespace(
        grad_data=grad_data,
        gradient_scaling_factor=1.0,
        params_list=[param],
    )
    bg = _bucket_group(buckets=[bucket])

    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_coalescing_manager",
        lambda group, async_ops: nullcontext(SimpleNamespace()),
    )
    monkeypatch.setattr(
        torch.distributed,
        "reduce_scatter_tensor",
        lambda *_args, **_kwargs: order.append("reduce-scatter"),
    )
    monkeypatch.setattr(rccl_sdma_param_gather, "get_sdma_process_group", lambda _g: SimpleNamespace())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: ConsumerStream())

    wrapped = grad_patches.make_start_grad_sync(lambda *_a, **_k: pytest.fail("fallback must not run"))
    wrapped(bg)

    assert order == ["wait", "reduce-scatter"]


def test_start_grad_sync_copies_extra_main_grads_before_scaling_and_collective(
    monkeypatch,
):
    observed = []
    grad_data = torch.zeros(8)
    rccl_sdma_param_gather.mark_direct_param_buffer(grad_data)
    param = SimpleNamespace(
        main_grad=torch.full((4,), 3.0),
        main_grad_copy_in_grad_buffer=grad_data[:4],
    )
    bucket = SimpleNamespace(
        grad_data=grad_data,
        gradient_scaling_factor=2.0,
        params_with_extra_main_grads=[param],
    )
    bg = _bucket_group(buckets=[bucket])

    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_coalescing_manager",
        lambda group, async_ops: nullcontext(SimpleNamespace()),
    )

    def reduce_scatter(*_args, **_kwargs):
        observed.append(grad_data.clone())

    monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", reduce_scatter)
    monkeypatch.setattr(rccl_sdma_param_gather, "get_sdma_process_group", lambda _g: SimpleNamespace())

    wrapped = grad_patches.make_start_grad_sync(lambda *_a, **_k: pytest.fail("fallback must not run"))
    wrapped(bg)

    assert len(observed) == 1
    assert torch.equal(observed[0][:4], torch.full((4,), 6.0))
    assert torch.equal(observed[0][4:], torch.zeros(4))


def test_start_grad_sync_sync_path_waits_and_synchronizes(monkeypatch):
    wait_calls = []
    sync_calls = []

    class NativeHandle:
        def wait(self):
            wait_calls.append(True)

    class ConsumerStream:
        def synchronize(self):
            sync_calls.append(True)

    grad_data = torch.zeros(8)
    rccl_sdma_param_gather.mark_direct_param_buffer(grad_data)
    bucket = SimpleNamespace(grad_data=grad_data, gradient_scaling_factor=1.0)
    bg = _bucket_group(buckets=[bucket])
    bg.ddp_config.overlap_grad_reduce = False

    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_coalescing_manager",
        lambda group, async_ops: nullcontext(NativeHandle()),
    )
    monkeypatch.setattr(
        torch.distributed,
        "reduce_scatter_tensor",
        lambda output, input_, op, group, async_op: None,
    )
    monkeypatch.setattr(rccl_sdma_param_gather, "get_sdma_process_group", lambda _g: SimpleNamespace())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: ConsumerStream())

    wrapped = grad_patches.make_start_grad_sync(lambda *_a, **_k: pytest.fail("fallback must not run"))
    wrapped(bg)

    assert wait_calls == [True]
    assert sync_calls == [True]
    assert bg.grad_reduce_handle is None


@pytest.mark.parametrize(
    "override",
    [
        {"force_all_reduce": True},
    ],
)
def test_start_grad_sync_falls_back_for_force_all_reduce(monkeypatch, override):
    grad_data = torch.zeros(8)
    rccl_sdma_param_gather.mark_direct_param_buffer(grad_data)
    bucket = SimpleNamespace(grad_data=grad_data, gradient_scaling_factor=1.0)
    bg = _bucket_group(buckets=[bucket])

    fallback_calls = []

    def fallback(self, force_all_reduce=False):
        fallback_calls.append((self, force_all_reduce))

    wrapped = grad_patches.make_start_grad_sync(fallback)
    wrapped(bg, **override)

    assert fallback_calls == [(bg, True)]


def test_start_grad_sync_falls_back_for_multi_instance(monkeypatch):
    grad_data = torch.zeros(8)
    rccl_sdma_param_gather.mark_direct_param_buffer(grad_data)
    bucket = SimpleNamespace(grad_data=grad_data, gradient_scaling_factor=1.0)
    bg = _bucket_group(buckets=[bucket])
    bg.ddp_config.num_distributed_optimizer_instances = 2

    fallback_calls = []
    wrapped = grad_patches.make_start_grad_sync(
        lambda self, force_all_reduce=False: fallback_calls.append(force_all_reduce)
    )
    wrapped(bg)

    assert fallback_calls == [False]


def test_start_grad_sync_falls_back_for_fp32_accumulation(monkeypatch):
    grad_data = torch.zeros(8)
    rccl_sdma_param_gather.mark_direct_param_buffer(grad_data)
    bucket = SimpleNamespace(grad_data=grad_data, gradient_scaling_factor=1.0)
    bg = _bucket_group(buckets=[bucket])
    bg.ddp_config.reduce_scatter_with_fp32_accumulation = True

    fallback_calls = []
    wrapped = grad_patches.make_start_grad_sync(
        lambda self, force_all_reduce=False: fallback_calls.append(True)
    )
    wrapped(bg)

    assert fallback_calls == [True]


def test_start_grad_sync_falls_back_when_not_all_buckets_direct(monkeypatch):
    direct_grad = torch.zeros(8)
    rccl_sdma_param_gather.mark_direct_param_buffer(direct_grad)
    plain_grad = torch.zeros(8)  # never marked -- e.g. the grad-alloc wrap did not run
    bg = _bucket_group(
        buckets=[
            SimpleNamespace(grad_data=direct_grad, gradient_scaling_factor=1.0),
            SimpleNamespace(grad_data=plain_grad, gradient_scaling_factor=1.0),
        ]
    )

    fallback_calls = []
    wrapped = grad_patches.make_start_grad_sync(
        lambda self, force_all_reduce=False: fallback_calls.append(True)
    )
    wrapped(bg)

    assert fallback_calls == [True]


def test_start_grad_sync_falls_back_when_not_distributed_optimizer(monkeypatch):
    plain_grad = torch.zeros(8)
    bg = _bucket_group(buckets=[SimpleNamespace(grad_data=plain_grad, gradient_scaling_factor=1.0)])
    bg.ddp_config.use_distributed_optimizer = False

    fallback_calls = []
    wrapped = grad_patches.make_start_grad_sync(
        lambda self, force_all_reduce=False: fallback_calls.append(True)
    )
    wrapped(bg)

    assert fallback_calls == [True]


def test_start_grad_sync_respects_first_batch_no_op(monkeypatch):
    grad_data = torch.zeros(8)
    rccl_sdma_param_gather.mark_direct_param_buffer(grad_data)
    bucket = SimpleNamespace(grad_data=grad_data, gradient_scaling_factor=1.0)
    bg = _bucket_group(buckets=[bucket], is_first_batch=True, grad_reduce_handle=SimpleNamespace())

    def boom(*_a, **_k):
        pytest.fail("must not issue a new collective while one is outstanding on first batch")

    monkeypatch.setattr(torch.distributed.distributed_c10d, "_coalescing_manager", boom)

    wrapped = grad_patches.make_start_grad_sync(
        lambda self, force_all_reduce=False: pytest.fail("fallback must not run either")
    )
    # Must be a true no-op: returns without touching grad_reduce_handle or grad_data.
    wrapped(bg)


def test_grad_buffer_wrapper_allocates_grad_data_from_pool_and_marks_buckets(
    monkeypatch,
):
    group = SimpleNamespace(group_name="ce")
    pool = SimpleNamespace()
    handle = SimpleNamespace()
    param_data = SimpleNamespace()
    grad_data = SimpleNamespace()
    bucket_grad_data = SimpleNamespace()
    pool_active = False
    allocation_scopes = []

    class PoolContext:
        def __enter__(self):
            nonlocal pool_active
            pool_active = True

        def __exit__(self, *_args):
            nonlocal pool_active
            pool_active = False

    def fake_real_zeros(*_args, **_kwargs):
        allocation_scopes.append(pool_active)
        return grad_data

    monkeypatch.setattr(grad_patches, "_REAL_TORCH_ZEROS", fake_real_zeros)
    monkeypatch.setattr(
        rccl_sdma_param_gather,
        "prepare_direct_param_buffer_pool",
        lambda _group, _device: (group, pool),
    )
    monkeypatch.setattr(
        rccl_sdma_param_gather,
        "rendezvous_direct_param_buffer",
        lambda tensor, _group: (rccl_sdma_param_gather.mark_direct_param_buffer(tensor) or handle),
    )
    monkeypatch.setattr(torch.cuda, "use_mem_pool", lambda _pool: PoolContext())

    def original(
        self,
        ddp_config,
        param_dtype,
        grad_dtype,
        params,
        data_parallel_group,
        bucket_size,
        param_to_name,
        gradient_scaling_factor,
        param_indices,
        nccl_ub,
        pg_collection=None,
    ):
        # Simulates the ALREADY param-patched __init__: param_data is already a
        # direct buffer by the time this (inner) original runs; only grad_data's
        # torch.zeros() call is live for this wrap to intercept.
        self.param_data = param_data
        self.grad_data = torch.zeros(1, dtype=grad_dtype, device="cuda")
        self.buckets = [SimpleNamespace(grad_data=bucket_grad_data)]

    wrapped = grad_patches.make_grad_and_param_buffer_init(original)
    buffer = SimpleNamespace()
    wrapped(
        buffer,
        SimpleNamespace(use_distributed_optimizer=True),
        torch.bfloat16,
        torch.bfloat16,
        [SimpleNamespace(device=torch.device("cuda", 0))],
        SimpleNamespace(),
        1024,
        {},
        1.0,
        [0],
        False,
    )

    assert buffer.grad_data is grad_data
    assert buffer._primus_rccl_sdma_grad_symmetric_memory is handle
    assert allocation_scopes == [True]
    assert rccl_sdma_param_gather.is_direct_param_buffer(grad_data)
    assert rccl_sdma_param_gather.is_direct_param_buffer(bucket_grad_data)
    # param_data (already handled by the inner/param wrap) must be untouched.
    assert buffer.param_data is param_data


def test_param_and_grad_allocation_wrappers_compose_without_intercepting_each_other(
    monkeypatch,
):
    """The inner param wrapper owns allocation 1; the outer grad wrapper owns 2."""
    group = SimpleNamespace(group_name="ce")
    pool = SimpleNamespace()
    handles = [SimpleNamespace(), SimpleNamespace()]
    allocations = []
    allocation_scopes = []
    pool_active = False
    real_torch_zeros = torch.zeros

    class PoolContext:
        def __enter__(self):
            nonlocal pool_active
            assert pool_active is False
            pool_active = True

        def __exit__(self, *_args):
            nonlocal pool_active
            pool_active = False

    def fake_real_zeros(*zeros_args, **zeros_kwargs):
        allocation_scopes.append(pool_active)
        tensor = real_torch_zeros(*zeros_args, **{**zeros_kwargs, "device": "cpu"})
        allocations.append(tensor)
        return tensor

    monkeypatch.setattr(param_patches, "_REAL_TORCH_ZEROS", fake_real_zeros)
    monkeypatch.setattr(grad_patches, "_REAL_TORCH_ZEROS", fake_real_zeros)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "use_mem_pool", lambda _pool: PoolContext())
    monkeypatch.setattr(
        rccl_sdma_param_gather,
        "prepare_direct_param_buffer_pool",
        lambda _group, _device: (group, pool),
    )
    monkeypatch.setattr(
        rccl_sdma_param_gather,
        "take_direct_param_buffer",
        lambda *_args, **_kwargs: None,
    )

    def rendezvous(tensor, _group):
        rccl_sdma_param_gather.mark_direct_param_buffer(tensor)
        return handles.pop(0)

    monkeypatch.setattr(rccl_sdma_param_gather, "rendezvous_direct_param_buffer", rendezvous)

    def original(
        self,
        ddp_config,
        param_dtype,
        grad_dtype,
        params,
        data_parallel_group,
        bucket_size,
        param_to_name,
        gradient_scaling_factor,
        param_indices,
        nccl_ub,
        pg_collection=None,
    ):
        del (
            bucket_size,
            param_to_name,
            gradient_scaling_factor,
            param_indices,
            pg_collection,
        )
        self.param_data = torch.zeros(4, dtype=param_dtype, device="cuda")
        self.grad_data = torch.zeros(4, dtype=grad_dtype, device="cuda")
        self.buckets = [SimpleNamespace(param_data=self.param_data[:], grad_data=self.grad_data[:])]

    inner = param_patches.make_param_and_grad_buffer_init(original)
    wrapped = grad_patches.make_grad_and_param_buffer_init(inner)
    buffer = SimpleNamespace()
    wrapped(
        buffer,
        SimpleNamespace(use_distributed_optimizer=True),
        torch.bfloat16,
        torch.bfloat16,
        [torch.nn.Parameter(torch.ones(1))],
        SimpleNamespace(),
        1024,
        {},
        1.0,
        [0],
        False,
    )

    assert buffer.param_data is allocations[0]
    assert buffer.grad_data is allocations[1]
    assert allocation_scopes == [True, True]
    assert rccl_sdma_param_gather.is_direct_param_buffer(buffer.param_data)
    assert rccl_sdma_param_gather.is_direct_param_buffer(buffer.grad_data)


def test_grad_buffer_wrapper_skips_when_not_distributed_optimizer(monkeypatch):
    calls = []

    def original(
        self,
        ddp_config,
        param_dtype,
        grad_dtype,
        params,
        data_parallel_group,
        bucket_size,
        param_to_name,
        gradient_scaling_factor,
        param_indices,
        nccl_ub,
        pg_collection=None,
    ):
        calls.append("original")
        self.param_data = None
        self.grad_data = None
        self.buckets = []

    wrapped = grad_patches.make_grad_and_param_buffer_init(original)
    buffer = SimpleNamespace()
    wrapped(
        buffer,
        SimpleNamespace(use_distributed_optimizer=False),
        torch.bfloat16,
        torch.bfloat16,
        [SimpleNamespace(device=torch.device("cuda", 0))],
        SimpleNamespace(),
        1024,
        {},
        1.0,
        [0],
        False,
    )

    assert calls == ["original"]
    assert not hasattr(buffer, "_primus_rccl_sdma_grad_symmetric_memory")


def test_grad_buffer_wrapper_skips_mxfp8_shared_buffer_path(monkeypatch):
    """The MXFP8 shared-buffer branch aliases grad_data to param_data and never
    calls torch.zeros a second time; this wrap must not misfire on it."""
    group = SimpleNamespace(group_name="ce")
    pool = SimpleNamespace()
    shared = SimpleNamespace()

    monkeypatch.setattr(
        rccl_sdma_param_gather,
        "prepare_direct_param_buffer_pool",
        lambda _group, _device: (group, pool),
    )

    def rendezvous_boom(*_a, **_k):
        pytest.fail("must not rendezvous grad_data when it was never freshly allocated here")

    monkeypatch.setattr(rccl_sdma_param_gather, "rendezvous_direct_param_buffer", rendezvous_boom)

    def original(
        self,
        ddp_config,
        param_dtype,
        grad_dtype,
        params,
        data_parallel_group,
        bucket_size,
        param_to_name,
        gradient_scaling_factor,
        param_indices,
        nccl_ub,
        pg_collection=None,
    ):
        # No torch.zeros call reaches this wrap's interceptor at all (e.g. the
        # shared buffer already existed / was allocated by an earlier wrap).
        self.param_data = shared
        self.grad_data = shared
        self.buckets = []

    wrapped = grad_patches.make_grad_and_param_buffer_init(original)
    buffer = SimpleNamespace()
    wrapped(
        buffer,
        SimpleNamespace(use_distributed_optimizer=True),
        torch.bfloat16,
        torch.bfloat16,
        [SimpleNamespace(device=torch.device("cuda", 0))],
        SimpleNamespace(),
        1024,
        {},
        1.0,
        [0],
        False,
    )

    assert buffer.grad_data is shared


def test_patch_registration_orders_after_param_gather_patch():
    from primus.backends.megatron.patches.parallelism import (
        rccl_sdma_param_all_gather_patches as param_patches,
    )
    from primus.core.patches.patch_registry import PatchRegistry

    grad_patch = PatchRegistry.get("megatron.distributed.rccl_sdma_grad_reduce_scatter")
    param_patch = PatchRegistry.get("megatron.distributed.rccl_sdma_param_all_gather")

    assert grad_patch is not None
    assert param_patch is not None
    assert grad_patch.priority > param_patch.priority
    assert grad_patch.condition is param_patches.rccl_sdma_param_gather_enabled
