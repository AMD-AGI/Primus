###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for HummingbirdXTArgBuilder.

``_load_hummingbirdxt_default_args`` (which imports the external ``train`` module)
is patched out so the pure merge/convert logic is testable without that package.
"""

from types import SimpleNamespace
from unittest import mock

import primus.backends.hummingbirdxt.argument_builder as ab


def _builder(defaults=None):
    with mock.patch.object(
        ab, "_load_hummingbirdxt_default_args", return_value=dict(defaults or {"lr": 0.1})
    ):
        return ab.HummingbirdXTArgBuilder()


def test_init_seeds_from_defaults():
    b = _builder({"lr": 0.1, "steps": 10})
    d = b.to_dict()
    assert d["lr"] == 0.1 and d["steps"] == 10


def test_update_deep_merges_over_defaults_and_chains():
    b = _builder({"lr": 0.1, "model": {"layers": 2}})
    ret = b.update(SimpleNamespace(lr=0.2, model=SimpleNamespace(hidden=8)))
    assert ret is b
    d = b.to_dict()
    assert d["lr"] == 0.2  # override
    assert d["model"] == {"layers": 2, "hidden": 8}  # deep merge


def test_to_dict_is_deep_copy():
    b = _builder({"m": {"k": 1}})
    b.to_dict()["m"]["k"] = 999
    assert b.to_dict()["m"]["k"] == 1


def test_to_namespace_and_finalize():
    b = _builder({})
    b.update(SimpleNamespace(model=SimpleNamespace(layers=4)))
    assert b.finalize().model.layers == 4
