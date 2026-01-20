###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for primus.backends.torchtitan.config_utils

Tests cover:
    - namespace_to_dict: SimpleNamespace → dict conversion
    - merge_dataclass_configs: Merging custom dataclass extensions
    - dict_to_dataclass: dict → dataclass conversion with dynamic fields
    - build_job_config_from_namespace: End-to-end JobConfig construction
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

# -----------------------------------------------------------------------------
# Test namespace_to_dict
# -----------------------------------------------------------------------------


def test_namespace_to_dict_simple():
    """Test converting a flat SimpleNamespace to dict."""
    from primus.backends.torchtitan.config_utils import namespace_to_dict

    ns = SimpleNamespace(a=1, b="hello", c=3.14)
    result = namespace_to_dict(ns)

    assert result == {"a": 1, "b": "hello", "c": 3.14}
    assert isinstance(result, dict)


def test_namespace_to_dict_nested():
    """Test converting nested SimpleNamespace to dict."""
    from primus.backends.torchtitan.config_utils import namespace_to_dict

    ns = SimpleNamespace(
        outer=SimpleNamespace(inner=SimpleNamespace(value=42), flag=True),
        top_level="test",
    )
    result = namespace_to_dict(ns)

    expected = {"outer": {"inner": {"value": 42}, "flag": True}, "top_level": "test"}
    assert result == expected


def test_namespace_to_dict_primitives():
    """Test that primitives are returned as-is."""
    from primus.backends.torchtitan.config_utils import namespace_to_dict

    assert namespace_to_dict(42) == 42
    assert namespace_to_dict("string") == "string"
    assert namespace_to_dict(3.14) == 3.14
    assert namespace_to_dict(None) is None


def test_namespace_to_dict_mixed():
    """Test converting SimpleNamespace with mixed types."""
    from primus.backends.torchtitan.config_utils import namespace_to_dict

    ns = SimpleNamespace(
        number=123,
        text="test",
        nested=SimpleNamespace(inner=456),
        list_val=[1, 2, 3],
        dict_val={"key": "value"},
    )
    result = namespace_to_dict(ns)

    assert result["number"] == 123
    assert result["text"] == "test"
    assert result["nested"] == {"inner": 456}
    assert result["list_val"] == [1, 2, 3]
    assert result["dict_val"] == {"key": "value"}


# -----------------------------------------------------------------------------
# Test merge_dataclass_configs
# -----------------------------------------------------------------------------


def test_merge_dataclass_configs_basic():
    """Test merging two simple dataclasses."""
    from primus.backends.torchtitan.config_utils import merge_dataclass_configs

    @dataclass
    class Base:
        field1: int = 1
        field2: str = "base"

    @dataclass
    class Custom:
        field2: str = "custom"  # Override
        field3: float = 3.14  # New field

    merged_cls = merge_dataclass_configs(Base, Custom)
    instance = merged_cls()

    # Check that merged class has all fields
    assert hasattr(instance, "field1")
    assert hasattr(instance, "field2")
    assert hasattr(instance, "field3")

    # Custom field2 should override
    assert instance.field2 == "custom"
    assert instance.field3 == 3.14


def test_merge_dataclass_configs_nested():
    """Test merging dataclasses with nested dataclass fields."""
    from primus.backends.torchtitan.config_utils import merge_dataclass_configs

    @dataclass
    class NestedBase:
        value: int = 10

    @dataclass
    class Base:
        nested: NestedBase = field(default_factory=NestedBase)
        other: str = "base"

    @dataclass
    class NestedCustom:
        value: int = 20
        extra: str = "custom"

    @dataclass
    class Custom:
        nested: NestedCustom = field(default_factory=NestedCustom)

    merged_cls = merge_dataclass_configs(Base, Custom)
    instance = merged_cls()

    # Should have both base and custom fields
    assert hasattr(instance, "nested")
    assert hasattr(instance, "other")

    # Nested should be merged
    assert hasattr(instance.nested, "value")
    assert hasattr(instance.nested, "extra")


def test_merge_dataclass_configs_no_overlap():
    """Test merging dataclasses with no overlapping fields."""
    from primus.backends.torchtitan.config_utils import merge_dataclass_configs

    @dataclass
    class Base:
        field_a: int = 1

    @dataclass
    class Custom:
        field_b: str = "custom"

    merged_cls = merge_dataclass_configs(Base, Custom)
    instance = merged_cls()

    assert hasattr(instance, "field_a")
    assert hasattr(instance, "field_b")
    assert instance.field_a == 1
    assert instance.field_b == "custom"


# -----------------------------------------------------------------------------
# Test dict_to_dataclass
# -----------------------------------------------------------------------------


def test_dict_to_dataclass_simple():
    """Test converting a simple dict to dataclass."""
    from primus.backends.torchtitan.config_utils import dict_to_dataclass

    @dataclass
    class TestClass:
        field1: int
        field2: str

    data = {"field1": 42, "field2": "test"}
    result = dict_to_dataclass(TestClass, data)

    assert isinstance(result, TestClass)
    assert result.field1 == 42
    assert result.field2 == "test"


def test_dict_to_dataclass_nested():
    """Test converting nested dicts to nested dataclasses."""
    from primus.backends.torchtitan.config_utils import dict_to_dataclass

    @dataclass
    class Inner:
        value: int

    @dataclass
    class Outer:
        inner: Inner
        other: str

    data = {"inner": {"value": 123}, "other": "text"}
    result = dict_to_dataclass(Outer, data)

    assert isinstance(result, Outer)
    assert isinstance(result.inner, Inner)
    assert result.inner.value == 123
    assert result.other == "text"


def test_dict_to_dataclass_dynamic_fields():
    """Test that unknown fields are attached dynamically."""
    from primus.backends.torchtitan.config_utils import dict_to_dataclass

    @dataclass
    class TestClass:
        known_field: int

    data = {"known_field": 42, "unknown_field": "extra", "another_unknown": 3.14}
    result = dict_to_dataclass(TestClass, data)

    # Known field should be set normally
    assert result.known_field == 42

    # Unknown fields should be attached dynamically
    assert hasattr(result, "unknown_field")
    assert result.unknown_field == "extra"
    assert hasattr(result, "another_unknown")
    assert result.another_unknown == 3.14


def test_dict_to_dataclass_partial():
    """Test converting dict with missing optional fields."""
    from primus.backends.torchtitan.config_utils import dict_to_dataclass

    @dataclass
    class TestClass:
        required: int
        optional: str = "default"

    data = {"required": 42}
    result = dict_to_dataclass(TestClass, data)

    assert result.required == 42
    assert result.optional == "default"


def test_dict_to_dataclass_non_dataclass():
    """Test that non-dataclass inputs are returned as-is."""
    from primus.backends.torchtitan.config_utils import dict_to_dataclass

    data = {"key": "value"}
    result = dict_to_dataclass(dict, data)

    assert result is data


# -----------------------------------------------------------------------------
# Test build_job_config_from_namespace (integration tests)
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_torchtitan_modules(monkeypatch):
    """Mock TorchTitan modules for testing."""

    @dataclass
    class MockExperimental:
        custom_args_module: str = ""

    @dataclass
    class MockModel:
        name: str = "llama3"
        flavor: str = "debugmodel"

    @dataclass
    class MockTraining:
        steps: int = 100
        global_batch_size: int = 128

    @dataclass
    class MockJobConfig:
        model: MockModel = field(default_factory=MockModel)
        training: MockTraining = field(default_factory=MockTraining)
        experimental: MockExperimental = field(default_factory=MockExperimental)

    # Mock the torchtitan.config.job_config module
    import sys
    from unittest.mock import MagicMock

    mock_module = MagicMock()
    mock_module.Experimental = MockExperimental
    mock_module.JobConfig = MockJobConfig
    sys.modules["torchtitan.config.job_config"] = mock_module
    sys.modules["torchtitan.config"] = MagicMock()
    sys.modules["torchtitan"] = MagicMock()

    yield {
        "Experimental": MockExperimental,
        "JobConfig": MockJobConfig,
        "Model": MockModel,
        "Training": MockTraining,
    }

    # Cleanup
    del sys.modules["torchtitan.config.job_config"]
    del sys.modules["torchtitan.config"]
    del sys.modules["torchtitan"]


def test_build_job_config_from_namespace_basic(mock_torchtitan_modules):
    """Test basic conversion from SimpleNamespace to JobConfig."""
    from primus.backends.torchtitan.config_utils import build_job_config_from_namespace

    ns = SimpleNamespace(
        model=SimpleNamespace(name="llama3", flavor="8B"),
        training=SimpleNamespace(steps=500, global_batch_size=256),
        experimental=SimpleNamespace(custom_args_module=""),
    )

    result = build_job_config_from_namespace(ns)

    assert result.model.name == "llama3"
    assert result.model.flavor == "8B"
    assert result.training.steps == 500
    assert result.training.global_batch_size == 256


def test_build_job_config_from_namespace_with_primus_config(mock_torchtitan_modules, monkeypatch):
    """Test that primus.* config is preserved and attached."""
    from primus.backends.torchtitan.config_utils import build_job_config_from_namespace

    # Mock log_rank_0 to avoid logger initialization issues
    monkeypatch.setattr("primus.backends.torchtitan.config_utils.log_rank_0", lambda msg: None)

    ns = SimpleNamespace(
        model=SimpleNamespace(name="llama3", flavor="8B"),
        training=SimpleNamespace(steps=100, global_batch_size=128),
        experimental=SimpleNamespace(custom_args_module=""),
        primus=SimpleNamespace(
            turbo=SimpleNamespace(enabled=True, fp8_format="e4m3"), custom_feature="value"
        ),
    )

    result = build_job_config_from_namespace(ns)

    # JobConfig fields should be set
    assert result.model.name == "llama3"
    assert result.training.steps == 100

    # Primus config should be attached
    assert hasattr(result, "primus")
    assert isinstance(result.primus, SimpleNamespace)
    assert result.primus.turbo.enabled is True
    assert result.primus.turbo.fp8_format == "e4m3"
    assert result.primus.custom_feature == "value"


def test_build_job_config_from_namespace_no_primus_config(mock_torchtitan_modules):
    """Test conversion without primus.* config."""
    from primus.backends.torchtitan.config_utils import build_job_config_from_namespace

    ns = SimpleNamespace(
        model=SimpleNamespace(name="llama3", flavor="debugmodel"),
        training=SimpleNamespace(steps=100, global_batch_size=128),
        experimental=SimpleNamespace(custom_args_module=""),
    )

    result = build_job_config_from_namespace(ns)

    # Should not have primus attribute if not provided
    assert not hasattr(result, "primus") or result.primus is None


def test_build_job_config_from_namespace_dynamic_fields(mock_torchtitan_modules):
    """Test that unknown fields are attached dynamically to JobConfig."""
    from primus.backends.torchtitan.config_utils import build_job_config_from_namespace

    ns = SimpleNamespace(
        model=SimpleNamespace(name="llama3", flavor="8B"),
        training=SimpleNamespace(steps=100, global_batch_size=128),
        experimental=SimpleNamespace(custom_args_module=""),
        unknown_top_level_field="test_value",
    )

    result = build_job_config_from_namespace(ns)

    # Known fields should work
    assert result.model.name == "llama3"

    # Unknown field should be attached
    assert hasattr(result, "unknown_top_level_field")
    assert result.unknown_top_level_field == "test_value"
