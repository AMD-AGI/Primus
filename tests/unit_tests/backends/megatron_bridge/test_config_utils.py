###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for primus.backends.megatron_bridge.config_utils

Tests cover:
    - namespace_to_dict: SimpleNamespace → dict conversion (including nested and mixed types)
"""

from types import SimpleNamespace

# -----------------------------------------------------------------------------
# Test namespace_to_dict
# -----------------------------------------------------------------------------


def test_namespace_to_dict_simple():
    """Test converting a flat SimpleNamespace to dict."""
    from primus.backends.megatron_bridge.config_utils import namespace_to_dict

    ns = SimpleNamespace(a=1, b="hello", c=3.14)
    result = namespace_to_dict(ns)

    assert result == {"a": 1, "b": "hello", "c": 3.14}
    assert isinstance(result, dict)


def test_namespace_to_dict_nested():
    """Test converting nested SimpleNamespace to dict."""
    from primus.backends.megatron_bridge.config_utils import namespace_to_dict

    ns = SimpleNamespace(
        outer=SimpleNamespace(inner=SimpleNamespace(value=42), flag=True),
        top_level="test",
    )
    result = namespace_to_dict(ns)

    expected = {"outer": {"inner": {"value": 42}, "flag": True}, "top_level": "test"}
    assert result == expected


def test_namespace_to_dict_primitives():
    """Test that primitives are returned as-is."""
    from primus.backends.megatron_bridge.config_utils import namespace_to_dict

    assert namespace_to_dict(42) == 42
    assert namespace_to_dict("string") == "string"
    assert namespace_to_dict(3.14) == 3.14
    assert namespace_to_dict(None) is None


def test_namespace_to_dict_with_lists():
    """Test converting SimpleNamespace with list containing namespaces."""
    from primus.backends.megatron_bridge.config_utils import namespace_to_dict

    ns = SimpleNamespace(
        items=[
            SimpleNamespace(id=1, name="first"),
            SimpleNamespace(id=2, name="second"),
        ],
        count=2,
    )
    result = namespace_to_dict(ns)

    assert result["count"] == 2
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 2
    assert result["items"][0] == {"id": 1, "name": "first"}
    assert result["items"][1] == {"id": 2, "name": "second"}


def test_namespace_to_dict_with_tuples():
    """Test converting SimpleNamespace with tuple containing namespaces."""
    from primus.backends.megatron_bridge.config_utils import namespace_to_dict

    ns = SimpleNamespace(
        coords=(SimpleNamespace(x=1, y=2), SimpleNamespace(x=3, y=4)),
    )
    result = namespace_to_dict(ns)

    assert isinstance(result["coords"], tuple)
    assert len(result["coords"]) == 2
    assert result["coords"][0] == {"x": 1, "y": 2}
    assert result["coords"][1] == {"x": 3, "y": 4}


def test_namespace_to_dict_with_dict():
    """Test converting SimpleNamespace containing regular dicts."""
    from primus.backends.megatron_bridge.config_utils import namespace_to_dict

    ns = SimpleNamespace(
        config={"key": "value", "nested": SimpleNamespace(inner="data")},
        flag=True,
    )
    result = namespace_to_dict(ns)

    assert result["flag"] is True
    assert isinstance(result["config"], dict)
    assert result["config"]["key"] == "value"
    assert result["config"]["nested"] == {"inner": "data"}


def test_namespace_to_dict_mixed():
    """Test converting SimpleNamespace with mixed types."""
    from primus.backends.megatron_bridge.config_utils import namespace_to_dict

    ns = SimpleNamespace(
        number=123,
        text="test",
        nested=SimpleNamespace(inner=456),
        list_val=[1, 2, 3],
        dict_val={"key": "value"},
        none_val=None,
    )
    result = namespace_to_dict(ns)

    assert result["number"] == 123
    assert result["text"] == "test"
    assert result["nested"] == {"inner": 456}
    assert result["list_val"] == [1, 2, 3]
    assert result["dict_val"] == {"key": "value"}
    assert result["none_val"] is None


def test_namespace_to_dict_empty():
    """Test converting an empty SimpleNamespace."""
    from primus.backends.megatron_bridge.config_utils import namespace_to_dict

    ns = SimpleNamespace()
    result = namespace_to_dict(ns)

    assert result == {}
    assert isinstance(result, dict)


def test_namespace_to_dict_deeply_nested():
    """Test converting deeply nested SimpleNamespace structures."""
    from primus.backends.megatron_bridge.config_utils import namespace_to_dict

    ns = SimpleNamespace(
        level1=SimpleNamespace(
            level2=SimpleNamespace(level3=SimpleNamespace(level4=SimpleNamespace(value="deep"), other=123))
        )
    )
    result = namespace_to_dict(ns)

    expected = {"level1": {"level2": {"level3": {"level4": {"value": "deep"}, "other": 123}}}}
    assert result == expected
