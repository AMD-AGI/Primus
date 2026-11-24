###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the MaxText source-rewrite helper (no JAX import; run on the torch CI image)."""

import sys
import types

import pytest

from primus.backends.maxtext.patches import _source_patch_utils as spu

_MODULE_SOURCE = """
def scale(x):
  y = x * 2
  return y


class Base:
  def value(self):
    return 1


class Layer(Base):
  def value(self):
    out = super().value()
    return out
"""


@pytest.fixture
def fake_module(tmp_path, monkeypatch):
    path = tmp_path / "fake_maxtext_module.py"
    path.write_text(_MODULE_SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("fake_maxtext_module", None)
    module = __import__("fake_maxtext_module")
    monkeypatch.setattr(spu, "SOURCE_PATCH_FAILURES", [])
    yield module
    sys.modules.pop("fake_maxtext_module", None)


def test_rewrite_module_function(fake_module):
    spu.rewrite_source(fake_module, "scale", [("  y = x * 2\n", "  y = x * 2\n  y += 1\n")])
    assert fake_module.scale(3) == 7


def test_rewrite_method_keeps_zero_arg_super(fake_module):
    spu.rewrite_source(fake_module.Layer, "value", [("    return out\n", "    return out + 10\n")])
    assert fake_module.Layer().value() == 11


def test_rewrites_stack_and_stay_detectable(fake_module):
    spu.rewrite_source(fake_module, "scale", [("  y = x * 2\n", "  y = x * 2\n  y += 1\n")])
    assert spu.any_in_source(fake_module, "scale", ["y += 1"])
    spu.rewrite_source(fake_module, "scale", [("  y += 1\n", "  y += 100\n")])
    assert fake_module.scale(3) == 106


def test_expected_count(fake_module):
    spu.rewrite_source(fake_module, "scale", [("y", "z", 2)])
    assert fake_module.scale(2) == 4


def test_anchor_drift_raises_and_is_recorded(fake_module):
    with pytest.raises(RuntimeError, match="Anchor matched 0 times"):
        spu.rewrite_source(fake_module, "scale", [("  y = x * 3\n", "")])
    assert len(spu.SOURCE_PATCH_FAILURES) == 1
    with pytest.raises(RuntimeError, match="could not be applied"):
        spu.raise_on_source_patch_failures()


def test_ambiguous_anchor_raises(fake_module):
    with pytest.raises(RuntimeError, match="Anchor matched 2 times"):
        spu.rewrite_source(fake_module, "scale", [("y", "z")])


def test_no_failures_is_a_no_op(monkeypatch):
    monkeypatch.setattr(spu, "SOURCE_PATCH_FAILURES", [])
    spu.raise_on_source_patch_failures()


def test_maxtext_patch_modules_import_without_maxtext():
    # Patch modules must register without importing MaxText/JAX at import time.
    import primus.backends.maxtext.patches  # noqa: F401
    from primus.core.patches.patch_registry import PatchRegistry

    ids = set(PatchRegistry.list_ids())
    for patch_id in (
        "maxtext.config.primus_fields",
        "maxtext.distributed.heartbeat_timeout",
        "maxtext.train.device_info",
        "maxtext.train.sync_before_loop",
        "maxtext.train.sync_each_step",
        "maxtext.moe.expert_balance",
        "maxtext.moe.routing_maps_remat",
        "maxtext.moe.turbo_grouped_gemm",
        "maxtext.attention.synthetic_causal_mask",
        "maxtext.models.query_pre_attn_scalar",
        "maxtext.lora.set_mesh_unbox",
    ):
        assert patch_id in ids
    assert isinstance(spu, types.ModuleType)
