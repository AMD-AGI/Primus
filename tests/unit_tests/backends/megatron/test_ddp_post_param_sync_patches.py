###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The DDP ``_post_param_sync`` fast path only replaces upstream's per-parameter walk."""

import sys
import types

import pytest

from primus.backends.megatron.patches.ddp_post_param_sync_patches import (
    _has_post_param_sync,
)


class _Core019Group:
    def _post_param_sync(self):
        quantized_params = []
        for bucket in self.buckets:
            for param in bucket.params:
                if _param_uses_quantized_storage(param):  # noqa: F821
                    quantized_params.append(param)
        if len(quantized_params) > 0:
            post_all_gather_processing(quantized_params)  # noqa: F821


class _MainGroup:
    def _post_param_sync(self):
        for bucket in self.buckets:
            for param in bucket.params:
                start, end = bucket.param_to_index[param]
                gathered_param = bucket.param_data[start:end].view_as(param)
                if param.data_ptr() != gathered_param.data_ptr():
                    param.data.copy_(gathered_param)


class _NoMethodGroup:
    pass


def _install(monkeypatch, group_cls):
    mod = types.ModuleType("megatron.core.distributed.param_and_grad_buffer")
    mod._ParamAndGradBucketGroup = group_cls
    for name in ("megatron", "megatron.core", "megatron.core.distributed"):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, name, pkg)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)


@pytest.mark.parametrize(
    "group_cls, expected",
    [(_MainGroup, True), (_Core019Group, False), (_NoMethodGroup, False)],
)
def test_condition_matches_upstream_walk(monkeypatch, group_cls, expected):
    _install(monkeypatch, group_cls)
    assert _has_post_param_sync(None) is expected
