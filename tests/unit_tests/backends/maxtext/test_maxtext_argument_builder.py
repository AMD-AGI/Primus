###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for MaxText argument builder helpers (no JAX import; run on the torch CI image)."""

import os
from types import SimpleNamespace

import yaml

from primus.backends.maxtext.argument_builder import (
    MaxTextConfigBuilder,
    export_params_to_yaml,
    namespace_to_dict,
)


def test_config_builder_update_finalize_roundtrip():
    b = MaxTextConfigBuilder()
    ns = SimpleNamespace(steps=3, model="llama")
    b.update(ns)
    assert b.finalize() is ns


def test_namespace_to_dict_recurses_nested_and_lists():
    ns = SimpleNamespace(a=1, nested=SimpleNamespace(b=2), lst=[SimpleNamespace(c=3), 4])
    assert namespace_to_dict(ns) == {"a": 1, "nested": {"b": 2}, "lst": [{"c": 3}, 4]}


def test_namespace_to_dict_passes_scalars_through():
    assert namespace_to_dict(5) == 5
    assert namespace_to_dict("x") == "x"
    assert namespace_to_dict({"k": SimpleNamespace(v=1)}) == {"k": {"v": 1}}


def test_export_params_to_yaml_writes_loadable_file():
    params = {"steps": 3, "model": "llama", "nested": {"lr": 0.1}}
    path = export_params_to_yaml(params)
    try:
        assert os.path.exists(path)
        with open(path) as f:
            assert yaml.safe_load(f) == params
    finally:
        os.remove(path)
