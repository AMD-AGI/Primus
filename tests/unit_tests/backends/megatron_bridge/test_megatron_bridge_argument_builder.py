###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for MegatronBridgeArgBuilder (pure config merge/convert).

Import is deferred into a fixture (like test_config_utils.py) so the package's
torch-pulling __init__ doesn't run at collection time under the coverage C tracer.
"""

from types import SimpleNamespace

import pytest


@pytest.fixture
def ArgBuilder():
    from primus.backends.megatron_bridge.argument_builder import (
        MegatronBridgeArgBuilder,
    )

    return MegatronBridgeArgBuilder


def test_update_deep_merges_and_chains(ArgBuilder):
    b = ArgBuilder()
    ret = b.update({"a": 1, "model": {"layers": 4}})
    assert ret is b
    b.update({"b": 2, "model": {"hidden": 8}})
    d = b.to_dict()
    assert d["a"] == 1 and d["b"] == 2
    # nested dict is deep-merged, not replaced
    assert d["model"] == {"layers": 4, "hidden": 8}


def test_update_accepts_namespace(ArgBuilder):
    b = ArgBuilder()
    b.update(SimpleNamespace(x=1, nested=SimpleNamespace(y=2)))
    d = b.to_dict()
    assert d["x"] == 1
    assert d["nested"]["y"] == 2


def test_later_update_overrides_scalar(ArgBuilder):
    b = ArgBuilder()
    b.update({"k": 1})
    b.update({"k": 2})
    assert b.to_dict()["k"] == 2


def test_to_dict_is_deep_copy(ArgBuilder):
    b = ArgBuilder()
    b.update({"m": {"k": 1}})
    snapshot = b.to_dict()
    snapshot["m"]["k"] = 999
    assert b.to_dict()["m"]["k"] == 1  # builder state isolated from returned dict


def test_finalize_returns_nested_namespace(ArgBuilder):
    b = ArgBuilder()
    b.update({"model": {"layers": 4}})
    assert b.finalize().model.layers == 4  # finalize aliases to_namespace
    assert b.to_namespace().model.layers == 4
