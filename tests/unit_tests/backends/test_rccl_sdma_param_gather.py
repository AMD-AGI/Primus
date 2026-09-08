###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.core.distributed import rccl_sdma_param_gather
from primus.backends.megatron.patches.parallelism import (
    rccl_sdma_param_all_gather_patches,
    sdma_param_all_gather_patches,
)

def test_rccl_backend_disables_direct_hip_patch(monkeypatch):
    monkeypatch.setenv("ENABLE_SDMA_ALLGATHER", "1")
    monkeypatch.setenv("MEGATRON_PARAM_GATHER_BACKEND", "rccl_sdma")

    assert rccl_sdma_param_all_gather_patches.rccl_sdma_param_gather_enabled()
    assert not sdma_param_all_gather_patches._sdma_allgather_enabled(None)


def test_direct_hip_patch_remains_available_without_rccl_backend(monkeypatch):
    monkeypatch.setenv("ENABLE_SDMA_ALLGATHER", "1")
    monkeypatch.delenv("MEGATRON_PARAM_GATHER_BACKEND", raising=False)

    assert not rccl_sdma_param_all_gather_patches.rccl_sdma_param_gather_enabled()
    assert sdma_param_all_gather_patches._sdma_allgather_enabled(None)

def test_direct_gather_uses_native_coalesced_work_handle(monkeypatch):
    monkeypatch.setenv("MEGATRON_PARAM_GATHER_BACKEND", "rccl_sdma")

    original_group = SimpleNamespace()
    dedicated_group = SimpleNamespace()
    native_handle = SimpleNamespace()
    coalescing_calls = []
    gather_calls = []

    def fake_coalescing_manager(group, async_ops):
        coalescing_calls.append((group, async_ops))
        return nullcontext(native_handle)

    def fake_all_gather(output, local_input, group, async_op):
        gather_calls.append((output, local_input, group, async_op))

    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_coalescing_manager",
        fake_coalescing_manager,
    )
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_all_gather)
    monkeypatch.setattr(
        rccl_sdma_param_gather,
        "get_sdma_process_group",
        lambda _group: dedicated_group,
    )
    param_data = torch.zeros(8)
    rccl_sdma_param_gather.mark_direct_param_buffer(param_data)
    bucket_group = SimpleNamespace(
        ddp_config=SimpleNamespace(
            use_distributed_optimizer=True,
            overlap_param_gather=True,
        ),
        param_gather_handle=None,
        cached_param_buffer_shard_list=[None],
        buckets=[SimpleNamespace(param_data=param_data)],
        intra_distributed_optimizer_instance_size=2,
        intra_distributed_optimizer_instance_rank=0,
        intra_distributed_optimizer_instance_group=original_group,
        param_gather_dispatched=False,
    )

    wrapped = rccl_sdma_param_all_gather_patches.make_start_param_sync(
        lambda *_args, **_kwargs: pytest.fail("native Megatron fallback must not run")
    )
    wrapped(bucket_group)

    assert coalescing_calls == [(dedicated_group, True)]
    assert len(gather_calls) == 1
    assert gather_calls[0][0] is param_data
    assert gather_calls[0][2:] == (dedicated_group, True)
    assert bucket_group.param_gather_handle is native_handle
    assert bucket_group.param_gather_dispatched


def test_direct_force_sync_uses_async_pg_stream_then_synchronizes(monkeypatch):
    monkeypatch.setenv("MEGATRON_PARAM_GATHER_BACKEND", "rccl_sdma")

    dedicated_group = SimpleNamespace()
    wait_calls = []
    sync_calls = []
    coalescing_calls = []
    gather_calls = []

    class NativeHandle:
        def wait(self):
            wait_calls.append(True)

    class ConsumerStream:
        def synchronize(self):
            sync_calls.append(True)

    native_handle = NativeHandle()

    def fake_coalescing_manager(group, async_ops):
        coalescing_calls.append((group, async_ops))
        return nullcontext(native_handle)

    def fake_all_gather(output, local_input, group, async_op):
        gather_calls.append((output, local_input, group, async_op))

    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_coalescing_manager",
        fake_coalescing_manager,
    )
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_all_gather)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device: ConsumerStream(),
    )
    monkeypatch.setattr(
        rccl_sdma_param_gather,
        "get_sdma_process_group",
        lambda _group: dedicated_group,
    )

    param_data = torch.zeros(8)
    rccl_sdma_param_gather.mark_direct_param_buffer(param_data)
    bucket_group = SimpleNamespace(
        ddp_config=SimpleNamespace(
            use_distributed_optimizer=True,
            overlap_param_gather=True,
        ),
        param_gather_handle=None,
        cached_param_buffer_shard_list=[None],
        buckets=[SimpleNamespace(param_data=param_data)],
        intra_distributed_optimizer_instance_size=2,
        intra_distributed_optimizer_instance_rank=0,
        intra_distributed_optimizer_instance_group=SimpleNamespace(),
        param_gather_dispatched=False,
    )

    wrapped = rccl_sdma_param_all_gather_patches.make_start_param_sync(
        lambda *_args, **_kwargs: pytest.fail("native Megatron fallback must not run")
    )
    wrapped(bucket_group, force_sync=True)

    assert coalescing_calls == [(dedicated_group, True)]
    assert len(gather_calls) == 1
    assert gather_calls[0][2:] == (dedicated_group, True)
    assert wait_calls == [True]
    assert sync_calls == [True]
    assert bucket_group.param_gather_handle is None
    assert bucket_group.param_gather_dispatched


def test_direct_buffer_marker_applies_to_views():
    tensor = SimpleNamespace()

    rccl_sdma_param_gather.mark_direct_param_buffer(tensor)

    assert rccl_sdma_param_gather.is_direct_param_buffer(tensor)


@pytest.mark.parametrize(
    ("requested", "recommended"),
    [
        (1, 2 * 1024 * 1024),
        (2 * 1024 * 1024, 2 * 1024 * 1024),
        (2 * 1024 * 1024 + 1, 4 * 1024 * 1024),
    ],
)
def test_recommended_eager_param_bytes(requested, recommended):
    assert (
        rccl_sdma_param_gather.recommended_eager_param_bytes(requested)
        == recommended
    )


def test_missing_eager_buffer_reports_retry_value(monkeypatch, capsys):
    device = torch.device("cuda", 0)
    group = SimpleNamespace(group_name="ce", rank=lambda: 0)
    original_empty = torch.empty

    def fake_empty(*args, **kwargs):
        if kwargs.get("device") == device:
            kwargs = dict(kwargs)
            kwargs.pop("device")
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fake_empty)
    rccl_sdma_param_gather.reset_runtime_state_for_tests()

    tensor = rccl_sdma_param_gather.take_direct_param_buffer(
        group,
        device,
        8,
        torch.bfloat16,
    )

    assert tensor is None
    output = capsys.readouterr().out
    assert "requested_bytes=16" in output
    assert "recommended_eager_bytes=2097152" in output
    assert "MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES=2097152" in output


def test_eager_direct_param_buffer_is_transferred_once(monkeypatch):
    device = torch.device("cuda", 0)
    group = SimpleNamespace(group_name="ce", rank=lambda: 0)
    pool = SimpleNamespace()
    original_empty = torch.empty

    def fake_empty(*args, **kwargs):
        if kwargs.get("device") == device:
            kwargs = dict(kwargs)
            kwargs.pop("device")
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(torch.cuda, "use_mem_pool", lambda _pool: nullcontext())
    rccl_sdma_param_gather.reset_runtime_state_for_tests()

    rccl_sdma_param_gather.reserve_direct_param_buffer(
        group,
        pool,
        device,
        16,
    )
    tensor = rccl_sdma_param_gather.take_direct_param_buffer(
        group,
        device,
        8,
        torch.bfloat16,
    )

    assert tensor is not None
    assert tensor.shape == (8,)
    assert tensor.dtype == torch.bfloat16
    assert tensor.count_nonzero() == 0
    assert (
        rccl_sdma_param_gather.take_direct_param_buffer(
            group,
            device,
            8,
            torch.bfloat16,
        )
        is None
    )


def test_param_buffer_wrapper_rendezvouses_and_marks_buckets(monkeypatch):
    import megatron.core.distributed.param_and_grad_buffer as pgb

    group = SimpleNamespace(group_name="ce", rank=lambda: 1)
    pool = SimpleNamespace()
    handle = SimpleNamespace()
    param_data = SimpleNamespace()
    grad_data = SimpleNamespace()
    bucket_data = SimpleNamespace()
    pool_active = False
    allocation_scopes = []

    class PoolContext:
        def __enter__(self):
            nonlocal pool_active
            pool_active = True

        def __exit__(self, *_args):
            nonlocal pool_active
            pool_active = False

    def fake_zeros(*_args, **_kwargs):
        allocation_scopes.append(pool_active)
        return param_data if len(allocation_scopes) == 1 else grad_data

    monkeypatch.setattr(pgb, "is_mxfp8tensor", lambda _param: False)
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
    monkeypatch.setattr(torch, "zeros", fake_zeros)
    monkeypatch.setattr(rccl_sdma_param_gather, "take_direct_param_buffer", lambda *_args: None)

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
            ddp_config,
            params,
            data_parallel_group,
            bucket_size,
            param_to_name,
            gradient_scaling_factor,
            param_indices,
            nccl_ub,
            pg_collection,
        )
        self.param_data = torch.zeros(1, dtype=param_dtype, device="cuda")
        self.grad_data = torch.zeros(1, dtype=grad_dtype, device="cuda")
        self.buckets = [SimpleNamespace(param_data=bucket_data)]

    wrapped = rccl_sdma_param_all_gather_patches.make_param_and_grad_buffer_init(original)
    buffer = SimpleNamespace()
    wrapped(
        buffer,
        SimpleNamespace(use_distributed_optimizer=True),
        torch.bfloat16,
        torch.float32,
        [SimpleNamespace(device=torch.device("cuda", 0))],
        SimpleNamespace(),
        1024,
        {},
        1.0,
        [0],
        False,
    )

    assert buffer._primus_rccl_sdma_pool is pool
    assert buffer._primus_rccl_sdma_symmetric_memory is handle
    assert buffer.grad_data is grad_data
    assert allocation_scopes == [True, False]
    assert rccl_sdma_param_gather.is_direct_param_buffer(param_data)
    assert rccl_sdma_param_gather.is_direct_param_buffer(bucket_data)


def test_param_buffer_allocation_failure_reports_eager_retry(monkeypatch):
    import megatron.core.distributed.param_and_grad_buffer as pgb

    device = torch.device("cuda", 0)
    group = SimpleNamespace(group_name="ce", rank=lambda: 0)
    pool = SimpleNamespace()

    monkeypatch.setattr(pgb, "is_mxfp8tensor", lambda _param: False)
    monkeypatch.setattr(
        rccl_sdma_param_gather,
        "prepare_direct_param_buffer_pool",
        lambda _group, _device: (group, pool),
    )
    monkeypatch.setattr(
        rccl_sdma_param_gather,
        "take_direct_param_buffer",
        lambda *_args: None,
    )
    monkeypatch.setattr(torch.cuda, "use_mem_pool", lambda _pool: nullcontext())

    def failing_zeros(*_args, **_kwargs):
        raise RuntimeError("ncclMemAlloc")

    monkeypatch.setattr(torch, "zeros", failing_zeros)

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
            ddp_config,
            grad_dtype,
            params,
            data_parallel_group,
            bucket_size,
            param_to_name,
            gradient_scaling_factor,
            param_indices,
            nccl_ub,
            pg_collection,
        )
        self.param_data = torch.zeros(8, dtype=param_dtype, device=device)

    wrapped = rccl_sdma_param_all_gather_patches.make_param_and_grad_buffer_init(
        original
    )
    with pytest.raises(
        RuntimeError,
        match=r"MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES=2097152",
    ):
        wrapped(
            SimpleNamespace(),
            SimpleNamespace(use_distributed_optimizer=True),
            torch.bfloat16,
            torch.float32,
            [SimpleNamespace(device=device)],
            SimpleNamespace(),
            1024,
            {},
            1.0,
            [0],
            False,
        )


@pytest.mark.parametrize(
    ("use_distributed_optimizer", "nccl_ub", "is_mxfp8", "error"),
    [
        (False, False, False, "requires the distributed optimizer"),
        (True, True, False, "does not support Megatron NCCL user buffers"),
        (True, False, True, "does not support MXFP8 parameter storage"),
    ],
)
def test_param_buffer_wrapper_rejects_unsupported_layouts(
    monkeypatch,
    use_distributed_optimizer,
    nccl_ub,
    is_mxfp8,
    error,
):
    import megatron.core.distributed.param_and_grad_buffer as pgb

    monkeypatch.setattr(pgb, "is_mxfp8tensor", lambda _param: is_mxfp8)

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
        pytest.fail("unsupported direct-gather layout reached Megatron allocation")

    wrapped = rccl_sdma_param_all_gather_patches.make_param_and_grad_buffer_init(original)
    with pytest.raises(RuntimeError, match=error):
        wrapped(
            SimpleNamespace(),
            SimpleNamespace(use_distributed_optimizer=use_distributed_optimizer),
            torch.bfloat16,
            torch.float32,
            [SimpleNamespace(device=torch.device("cuda", 0))],
            SimpleNamespace(),
            1024,
            {},
            1.0,
            [0],
            nccl_ub,
        )


def test_global_cta_policy_is_rejected_for_megatron_backend(monkeypatch):
    monkeypatch.setenv("NCCL_CTA_POLICY", "2")

    with pytest.raises(RuntimeError, match="requires NCCL_CTA_POLICY to be unset"):
        rccl_sdma_param_all_gather_patches.validate_global_cta_policy()


def test_dedicated_group_gets_zero_cta_without_mutating_original(monkeypatch):
    class OriginalGroup:
        def __init__(self):
            self.policy = "unchanged"

        def size(self):
            return 8

    class FakeOptions:
        def __init__(self):
            self.config = SimpleNamespace(cta_policy=None, split_share=None)

    class FakeProcessGroupNCCL:
        Options = FakeOptions
        NCCL_CTA_POLICY_ZERO = 2
        NCCL_CTA_POLICY_DEFAULT = 0

    captured = {}
    dedicated_group = SimpleNamespace(group_name="dedicated_ce_group")

    def fake_new_group(**kwargs):
        captured.update(kwargs)
        return dedicated_group

    original_group = OriginalGroup()
    monkeypatch.setattr(
        rccl_sdma_param_gather.dist,
        "ProcessGroupNCCL",
        FakeProcessGroupNCCL,
    )
    monkeypatch.setattr(rccl_sdma_param_gather.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(rccl_sdma_param_gather.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(rccl_sdma_param_gather.dist, "new_group", fake_new_group)
    monkeypatch.delenv("MEGATRON_RCCL_SDMA_CTA_POLICY", raising=False)
    rccl_sdma_param_gather.reset_runtime_state_for_tests()

    try:
        result = rccl_sdma_param_gather.get_sdma_process_group(original_group)
    finally:
        rccl_sdma_param_gather.reset_runtime_state_for_tests()

    assert result is dedicated_group
    assert captured["ranks"] == list(range(8))
    assert captured["backend"] == "nccl"
    assert captured["pg_options"].config.cta_policy == 2
    assert captured["pg_options"].config.split_share == 0
    assert "PARAM_GATHER_POLICY_2" in captured["group_desc"]
    assert original_group.policy == "unchanged"


def _run_sdma_hook(extra_env):
    hook = Path(__file__).resolve().parents[3] / "runner/helpers/hooks/06_enable_sdma_all_gather.sh"
    env = os.environ.copy()
    for name in (
        "FSDP_ALL_GATHER_BACKEND",
        "MEGATRON_PARAM_GATHER_BACKEND",
        "MEGATRON_RCCL_SDMA_DIRECT",
        "MEGATRON_RCCL_SDMA_SCRATCH_BYTES",
        "MEGATRON_RCCL_SDMA_EAGER_INIT",
        "NCCL_CTA_POLICY",
        "NCCL_CUMEM_ENABLE",
    ):
        env.pop(name, None)
    env.update(extra_env)
    return subprocess.run(
        ["bash", str(hook)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_megatron_hook_enables_cumem_without_global_cta_policy():
    result = _run_sdma_hook(
        {"MEGATRON_PARAM_GATHER_BACKEND": "rccl_sdma"},
    )

    assert result.returncode == 0
    assert "env.MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma" in result.stdout
    assert "env.NCCL_CTA_POLICY" not in result.stdout
    assert "env.NCCL_CUMEM_ENABLE=1" in result.stdout
    assert "env.MEGATRON_RCCL_SDMA_EAGER_INIT=1" in result.stdout
    assert "MEGATRON_RCCL_SDMA_DIRECT" not in result.stdout
    assert "MEGATRON_RCCL_SDMA_SCRATCH_BYTES" not in result.stdout


def test_fsdp_hook_keeps_global_cumem_and_cta_policy():
    result = _run_sdma_hook(
        {"FSDP_ALL_GATHER_BACKEND": "rccl_sdma"},
    )

    assert result.returncode == 0
    assert "env.NCCL_CUMEM_ENABLE=1" in result.stdout
    assert "env.NCCL_CTA_POLICY=2" in result.stdout


def test_hook_rejects_combined_fsdp_and_megatron_backends():
    result = _run_sdma_hook(
        {
            "FSDP_ALL_GATHER_BACKEND": "rccl_sdma",
            "MEGATRON_PARAM_GATHER_BACKEND": "rccl_sdma",
        },
    )

    assert result.returncode == 2
    assert "cannot be enabled together" in result.stderr
