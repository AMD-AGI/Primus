###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for primus.core.utils.yaml_utils (namespace <-> dict + merge helpers).

These pure helpers back the backend argument builders (dict_to_nested_namespace /
nested_namespace_to_dict / deep_merge); no torch/framework imports.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml

from primus.core.utils.yaml_utils import (
    check_key_in_namespace,
    deep_merge_namespace,
    delete_namespace_key,
    dict_to_nested_namespace,
    dump_namespace_to_yaml,
    get_value_by_key,
    has_key_in_namespace,
    merge_namespace,
    nested_namespace_to_dict,
    override_namespace,
    parse_nested_namespace_to_str,
    set_value_by_key,
)


def test_dict_to_nested_namespace_recurses_into_lists_and_maps():
    ns = dict_to_nested_namespace({"a": 1, "b": {"c": 2}, "l": [{"d": 3}, 4]})
    assert ns.a == 1
    assert ns.b.c == 2
    assert ns.l[0].d == 3 and ns.l[1] == 4


def test_dict_to_nested_namespace_passes_scalars_through():
    assert dict_to_nested_namespace(5) == 5


def test_nested_namespace_to_dict_roundtrips():
    d = {"a": 1, "b": {"c": 2}, "l": [{"d": 3}]}
    assert nested_namespace_to_dict(dict_to_nested_namespace(d)) == d


def test_key_set_get_has_delete():
    ns = SimpleNamespace(name="n")
    set_value_by_key(ns, "x", 1)
    assert has_key_in_namespace(ns, "x")
    assert get_value_by_key(ns, "x") == 1
    delete_namespace_key(ns, "x")
    assert not has_key_in_namespace(ns, "x")


def test_set_value_null_string_becomes_none():
    ns = SimpleNamespace()
    set_value_by_key(ns, "x", "null")
    assert ns.x is None


def test_set_value_rejects_override_unless_allowed():
    ns = SimpleNamespace(x=1)
    with pytest.raises(AssertionError):
        set_value_by_key(ns, "x", 2)
    set_value_by_key(ns, "x", 2, allow_override=True)
    assert ns.x == 2


def test_check_key_missing_raises():
    ns = SimpleNamespace(name="cfg")
    with pytest.raises(AssertionError, match="Failed to find key"):
        check_key_in_namespace(ns, "missing")


def test_deep_merge_namespace_inplace_preserves_nested_reference():
    ns = SimpleNamespace(a=1, sub=SimpleNamespace(x=1))
    sub_ref = ns.sub
    deep_merge_namespace(ns, {"a": 2, "sub": {"y": 3}})
    assert ns.a == 2
    assert ns.sub.x == 1 and ns.sub.y == 3
    assert ns.sub is sub_ref  # existing nested namespace updated in place


def test_override_namespace_deep_merges_and_handles_none():
    base = SimpleNamespace(a=1, b=SimpleNamespace(c=1))
    override_namespace(base, SimpleNamespace(b=SimpleNamespace(d=2), e=3))
    assert base.b.c == 1 and base.b.d == 2 and base.e == 3
    override_namespace(base, None)  # no-op
    assert base.a == 1


def test_merge_namespace_skips_existing_by_default():
    dst = SimpleNamespace(a=1)
    merge_namespace(dst, SimpleNamespace(a=2, b=3))
    assert dst.a == 1 and dst.b == 3  # existing 'a' kept, new 'b' added


def test_merge_namespace_allow_override_and_excepts():
    dst = SimpleNamespace(a=1)
    merge_namespace(dst, SimpleNamespace(a=2), allow_override=True)
    assert dst.a == 2

    dst2 = SimpleNamespace()
    merge_namespace(dst2, SimpleNamespace(a=1, secret=2), excepts=["secret"])
    assert dst2.a == 1 and not has_key_in_namespace(dst2, "secret")


def test_dump_namespace_to_yaml_roundtrip(tmp_path):
    ns = SimpleNamespace(a=1, b=SimpleNamespace(c=2), lst=[1, 2])
    path = tmp_path / "cfg.yaml"
    dump_namespace_to_yaml(ns, str(path))
    assert yaml.safe_load(path.read_text()) == {"a": 1, "b": {"c": 2}, "lst": [1, 2]}


def test_parse_nested_namespace_to_str_is_json():
    s = parse_nested_namespace_to_str(SimpleNamespace(a=1, b=SimpleNamespace(c=2)))
    assert json.loads(s) == {"a": 1, "b": {"c": 2}}
