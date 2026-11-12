###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the fused_pad_routing_map patch's sweep of stale references.

The sweep runs over every module in sys.modules. Images that ship apex in JIT
mode register amp_C, syncbn, ... there as proxies that build the extension in
``__getattr__``, so probing them costs ~17 s per op on a cold extension cache.
"""

import sys
from types import ModuleType

import pytest

pytest.importorskip("megatron")
pytest.importorskip("triton")

import megatron.core.fusions.fused_pad_routing_map as meg_mod

from primus.backends.megatron.core.fusions.fused_pad_routing_map import (
    fused_pad_routing_map as primus_fused_pad_routing_map,
)
from primus.backends.megatron.patches.fused_pad_routing_map_patches import (
    patch_fused_pad_routing_map,
)


class _ApexOpProxy:
    def __getattr__(self, name):
        self.__dict__.setdefault("probed", []).append(name)
        raise AttributeError(name)


def test_rebinds_importers_without_probing_proxies(monkeypatch):
    # A stand-in original, so the test neither depends on nor leaks the real patch state.
    stale = lambda *args, **kwargs: None
    monkeypatch.setattr(meg_mod, "fused_pad_routing_map", stale)
    importer = ModuleType("_primus_fprm_importer")
    importer.fused_pad_routing_map = stale
    proxy = _ApexOpProxy()
    monkeypatch.setitem(sys.modules, importer.__name__, importer)
    monkeypatch.setitem(sys.modules, "_primus_fprm_proxy", proxy)

    patch_fused_pad_routing_map(None)

    assert meg_mod.fused_pad_routing_map is primus_fused_pad_routing_map
    assert importer.fused_pad_routing_map is primus_fused_pad_routing_map
    assert "probed" not in proxy.__dict__


def test_skips_when_megatron_drops_the_symbol(monkeypatch):
    monkeypatch.delattr(meg_mod, "fused_pad_routing_map")

    patch_fused_pad_routing_map(None)

    assert not hasattr(meg_mod, "fused_pad_routing_map")
