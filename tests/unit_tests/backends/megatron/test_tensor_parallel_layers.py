###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Distributed forward coverage for LinearWithGradAccumulationAndAsyncCommunication (PRPUNDIT-16).

Existing trainer smokes run this patched TP-linear with TP=1, so they never all-gather
a real sequence-parallel shard. These tests use a 2-rank Gloo group and an independent
list-all_gather oracle.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

pytest.importorskip("megatron.core")

import primus.backends.megatron.core.tensor_parallel.layers as tp_layers
from primus.backends.megatron.core.tensor_parallel.layers import (
    LinearWithGradAccumulationAndAsyncCommunication,
)

_SEQ_PER_RANK = 4
_HIDDEN = 8
_OUT = 16


class _CpuMemoryBuffer:
    def get_tensor(self, dim_size, dtype, name):
        return torch.empty(dim_size, dtype=dtype)


def _independent_gathered_linear(local_input, weight, bias, group):
    shards = [torch.empty_like(local_input) for _ in range(dist.get_world_size(group))]
    dist.all_gather(shards, local_input.contiguous(), group=group)
    gathered = torch.cat(shards, dim=0)
    output = torch.matmul(gathered, weight.t())
    if bias is not None:
        output = output + bias
    return output


class TestLinearWithGradAccumulationAndAsyncCommunicationForward(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return 2

    def _init_gloo(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(backend="gloo", world_size=self.world_size, rank=self.rank, store=store)

    def _install_cpu_parallel_stubs(self):
        tp_layers.get_tensor_model_parallel_world_size = lambda: dist.get_world_size()
        buffer = _CpuMemoryBuffer()
        tp_layers.get_global_memory_buffer = lambda: buffer

    def _shared_weight_and_bias(self, use_bias):
        torch.manual_seed(0)
        if self.rank == 0:
            weight = torch.randn(_OUT, _HIDDEN)
            bias = torch.randn(_OUT) if use_bias else None
        else:
            weight = torch.empty(_OUT, _HIDDEN)
            bias = torch.empty(_OUT) if use_bias else None
        dist.broadcast(weight, src=0)
        if bias is not None:
            dist.broadcast(bias, src=0)
        weight.main_grad = torch.zeros_like(weight)
        return weight, bias

    def _local_input(self):
        # Distinct per-rank sequence shards so a skipped or reordered gather cannot hide.
        return torch.arange(_SEQ_PER_RANK * _HIDDEN, dtype=torch.float32).reshape(
            _SEQ_PER_RANK, _HIDDEN
        ) + float(self.rank * 1000)

    def _apply(self, local_input, weight, bias, sequence_parallel):
        return LinearWithGradAccumulationAndAsyncCommunication.apply(
            local_input,
            weight,
            bias,
            False,
            False,
            sequence_parallel,
            None,
            0,
            dist.group.WORLD,
        )

    def test_sequence_parallel_matches_independent_gathered_matmul_with_bias(self):
        self._init_gloo()
        self._install_cpu_parallel_stubs()
        weight, bias = self._shared_weight_and_bias(use_bias=True)
        local_input = self._local_input()
        got = self._apply(local_input, weight, bias, sequence_parallel=True)
        expected = _independent_gathered_linear(local_input, weight, bias, dist.group.WORLD)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(got.shape[0], _SEQ_PER_RANK * self.world_size)

    def test_sequence_parallel_matches_independent_gathered_matmul_without_bias(self):
        self._init_gloo()
        self._install_cpu_parallel_stubs()
        weight, bias = self._shared_weight_and_bias(use_bias=False)
        local_input = self._local_input()
        got = self._apply(local_input, weight, bias, sequence_parallel=True)
        expected = _independent_gathered_linear(local_input, weight, bias, dist.group.WORLD)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    def test_non_sequence_parallel_uses_local_input_and_skips_gather(self):
        self._init_gloo()
        self._install_cpu_parallel_stubs()
        gather_calls = []
        real_gather = tp_layers.dist_all_gather_func

        def _counting_gather(*args, **kwargs):
            gather_calls.append(SimpleNamespace(args=args, kwargs=kwargs))
            return real_gather(*args, **kwargs)

        tp_layers.dist_all_gather_func = _counting_gather
        weight, bias = self._shared_weight_and_bias(use_bias=True)
        local_input = self._local_input()
        got = self._apply(local_input, weight, bias, sequence_parallel=False)
        expected = torch.matmul(local_input, weight.t()) + bias
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(got.shape[0], _SEQ_PER_RANK)
        self.assertEqual(gather_calls, [])


if __name__ == "__main__":
    run_tests()
