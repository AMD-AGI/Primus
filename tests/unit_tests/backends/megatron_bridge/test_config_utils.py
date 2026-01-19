###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for primus.backends.megatron_bridge.config_utils

Tests cover:
    - namespace_to_dict: SimpleNamespace → dict conversion (including nested and mixed types)
    - _dict_to_dataclass: dict → dataclass conversion with dynamic fields (fallback)
    - build_job_config_from_namespace: End-to-end ConfigContainer construction
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

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


# -----------------------------------------------------------------------------
# Test _dict_to_dataclass (fallback implementation)
# -----------------------------------------------------------------------------


def test_dict_to_dataclass_simple():
    """Test converting a simple dict to dataclass."""
    from primus.backends.megatron_bridge.config_utils import _dict_to_dataclass

    @dataclass
    class TestClass:
        field1: int
        field2: str

    data = {"field1": 42, "field2": "test"}
    result = _dict_to_dataclass(TestClass, data)

    assert isinstance(result, TestClass)
    assert result.field1 == 42
    assert result.field2 == "test"


def test_dict_to_dataclass_nested():
    """Test converting nested dicts to nested dataclasses."""
    from primus.backends.megatron_bridge.config_utils import _dict_to_dataclass

    @dataclass
    class Inner:
        value: int

    @dataclass
    class Outer:
        inner: Inner
        other: str

    data = {"inner": {"value": 123}, "other": "text"}
    result = _dict_to_dataclass(Outer, data)

    assert isinstance(result, Outer)
    assert isinstance(result.inner, Inner)
    assert result.inner.value == 123
    assert result.other == "text"


def test_dict_to_dataclass_dynamic_fields():
    """Test that unknown fields are attached dynamically."""
    from primus.backends.megatron_bridge.config_utils import _dict_to_dataclass

    @dataclass
    class TestClass:
        known_field: int

    data = {"known_field": 42, "unknown_field": "extra", "another_unknown": 3.14}
    result = _dict_to_dataclass(TestClass, data)

    # Known field should be set normally
    assert result.known_field == 42

    # Unknown fields should be attached dynamically
    assert hasattr(result, "unknown_field")
    assert result.unknown_field == "extra"
    assert hasattr(result, "another_unknown")
    assert result.another_unknown == 3.14


def test_dict_to_dataclass_partial():
    """Test converting dict with missing optional fields."""
    from primus.backends.megatron_bridge.config_utils import _dict_to_dataclass

    @dataclass
    class TestClass:
        required: int
        optional: str = "default"

    data = {"required": 42}
    result = _dict_to_dataclass(TestClass, data)

    assert result.required == 42
    assert result.optional == "default"


def test_dict_to_dataclass_non_dataclass():
    """Test that non-dataclass inputs are returned as-is."""
    from primus.backends.megatron_bridge.config_utils import _dict_to_dataclass

    data = {"key": "value"}
    result = _dict_to_dataclass(dict, data)

    assert result is data


# -----------------------------------------------------------------------------
# Test build_job_config_from_namespace (integration tests)
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_megatron_bridge_modules(monkeypatch):
    """Mock Megatron-Bridge modules for testing."""

    @dataclass
    class MockTrainingConfig:
        num_steps: int = 1000
        global_batch_size: int = 32

    @dataclass
    class MockModelProvider:
        num_layers: int = 12
        hidden_size: int = 768

    @dataclass
    class MockOptimizerConfig:
        lr: float = 1e-4
        weight_decay: float = 0.01

    @dataclass
    class MockSchedulerConfig:
        warmup_steps: int = 100

    @dataclass
    class MockDatasetConfig:
        data_path: str = "/data/train"

    @dataclass
    class MockLoggerConfig:
        log_interval: int = 10

    @dataclass
    class MockTokenizerConfig:
        tokenizer_model: str = "gpt2"

    @dataclass
    class MockCheckpointConfig:
        save_interval: int = 1000

    @dataclass
    class MockConfigContainer:
        train: MockTrainingConfig = field(default_factory=MockTrainingConfig)
        model: MockModelProvider = field(default_factory=MockModelProvider)
        optimizer: MockOptimizerConfig = field(default_factory=MockOptimizerConfig)
        scheduler: MockSchedulerConfig = field(default_factory=MockSchedulerConfig)
        dataset: MockDatasetConfig = field(default_factory=MockDatasetConfig)
        logger: MockLoggerConfig = field(default_factory=MockLoggerConfig)
        tokenizer: MockTokenizerConfig = field(default_factory=MockTokenizerConfig)
        checkpoint: MockCheckpointConfig = field(default_factory=MockCheckpointConfig)

        @classmethod
        def from_dict(cls, config_dict, mode=None):
            """Mock from_dict implementation."""
            # Remove _target_ if present
            config_dict.pop("_target_", None)

            # Simple implementation that handles basic nested configs
            kwargs = {}
            for f in ["train", "model", "optimizer", "scheduler", "dataset", "logger", "tokenizer", "checkpoint"]:
                if f in config_dict and isinstance(config_dict[f], dict):
                    field_cls = {
                        "train": MockTrainingConfig,
                        "model": MockModelProvider,
                        "optimizer": MockOptimizerConfig,
                        "scheduler": MockSchedulerConfig,
                        "dataset": MockDatasetConfig,
                        "logger": MockLoggerConfig,
                        "tokenizer": MockTokenizerConfig,
                        "checkpoint": MockCheckpointConfig,
                    }[f]
                    kwargs[f] = field_cls(**config_dict[f])

            return cls(**kwargs)

    class MockInstantiationMode:
        STRICT = "strict"
        LENIENT = "lenient"

    # Mock the megatron.bridge modules
    import sys
    from unittest.mock import MagicMock

    # Mock megatron.bridge.training.config
    mock_config_module = MagicMock()
    mock_config_module.ConfigContainer = MockConfigContainer
    sys.modules["megatron.bridge.training.config"] = mock_config_module

    # Mock megatron.bridge.training.utils.config_utils
    mock_utils_module = MagicMock()
    mock_utils_module.InstantiationMode = MockInstantiationMode
    sys.modules["megatron.bridge.training.utils.config_utils"] = mock_utils_module

    # Mock parent modules
    sys.modules["megatron"] = MagicMock()
    sys.modules["megatron.bridge"] = MagicMock()
    sys.modules["megatron.bridge.training"] = MagicMock()
    sys.modules["megatron.bridge.training.utils"] = MagicMock()

    # Mock log_rank_0 to avoid logger initialization issues
    monkeypatch.setattr("primus.backends.megatron_bridge.config_utils.log_rank_0", lambda msg: None)

    yield {
        "ConfigContainer": MockConfigContainer,
        "InstantiationMode": MockInstantiationMode,
        "TrainingConfig": MockTrainingConfig,
        "ModelProvider": MockModelProvider,
    }

    # Cleanup
    for module in [
        "megatron.bridge.training.utils.config_utils",
        "megatron.bridge.training.config",
        "megatron.bridge.training.utils",
        "megatron.bridge.training",
        "megatron.bridge",
        "megatron",
    ]:
        if module in sys.modules:
            del sys.modules[module]


def test_build_job_config_from_namespace_basic(mock_megatron_bridge_modules):
    """Test basic conversion from SimpleNamespace to ConfigContainer."""
    from primus.backends.megatron_bridge.config_utils import build_job_config_from_namespace

    ns = SimpleNamespace(
        train=SimpleNamespace(num_steps=2000, global_batch_size=64),
        model=SimpleNamespace(num_layers=24, hidden_size=1024),
        optimizer=SimpleNamespace(lr=2e-4, weight_decay=0.1),
        scheduler=SimpleNamespace(warmup_steps=200),
        dataset=SimpleNamespace(data_path="/custom/data"),
        logger=SimpleNamespace(log_interval=20),
        tokenizer=SimpleNamespace(tokenizer_model="llama"),
        checkpoint=SimpleNamespace(save_interval=500),
    )

    result = build_job_config_from_namespace(ns)

    assert result.train.num_steps == 2000
    assert result.train.global_batch_size == 64
    assert result.model.num_layers == 24
    assert result.model.hidden_size == 1024
    assert result.optimizer.lr == 2e-4


def test_build_job_config_from_namespace_with_primus_config(mock_megatron_bridge_modules):
    """Test that primus.* config is preserved and attached."""
    from primus.backends.megatron_bridge.config_utils import build_job_config_from_namespace

    ns = SimpleNamespace(
        train=SimpleNamespace(num_steps=1000, global_batch_size=32),
        model=SimpleNamespace(num_layers=12, hidden_size=768),
        optimizer=SimpleNamespace(lr=1e-4, weight_decay=0.01),
        scheduler=SimpleNamespace(warmup_steps=100),
        dataset=SimpleNamespace(data_path="/data/train"),
        logger=SimpleNamespace(log_interval=10),
        tokenizer=SimpleNamespace(tokenizer_model="gpt2"),
        checkpoint=SimpleNamespace(save_interval=1000),
        primus=SimpleNamespace(
            workspace="/workspace",
            exp_name="test_experiment",
            custom_feature=SimpleNamespace(enabled=True, value=42),
        ),
    )

    result = build_job_config_from_namespace(ns)

    # ConfigContainer fields should be set
    assert result.train.num_steps == 1000
    assert result.model.num_layers == 12

    # Primus config should be attached
    assert hasattr(result, "primus")
    assert isinstance(result.primus, SimpleNamespace)
    assert result.primus.workspace == "/workspace"
    assert result.primus.exp_name == "test_experiment"
    assert result.primus.custom_feature.enabled is True
    assert result.primus.custom_feature.value == 42


def test_build_job_config_from_namespace_no_primus_config(mock_megatron_bridge_modules):
    """Test conversion without primus.* config."""
    from primus.backends.megatron_bridge.config_utils import build_job_config_from_namespace

    ns = SimpleNamespace(
        train=SimpleNamespace(num_steps=1000, global_batch_size=32),
        model=SimpleNamespace(num_layers=12, hidden_size=768),
        optimizer=SimpleNamespace(lr=1e-4, weight_decay=0.01),
        scheduler=SimpleNamespace(warmup_steps=100),
        dataset=SimpleNamespace(data_path="/data/train"),
        logger=SimpleNamespace(log_interval=10),
        tokenizer=SimpleNamespace(tokenizer_model="gpt2"),
        checkpoint=SimpleNamespace(save_interval=1000),
    )

    result = build_job_config_from_namespace(ns)

    # Should not have primus attribute if not provided
    assert not hasattr(result, "primus") or result.primus is None


def test_build_job_config_from_namespace_with_target_field(mock_megatron_bridge_modules):
    """Test that _target_ field is automatically added."""
    from primus.backends.megatron_bridge.config_utils import build_job_config_from_namespace

    ns = SimpleNamespace(
        train=SimpleNamespace(num_steps=1000, global_batch_size=32),
        model=SimpleNamespace(num_layers=12, hidden_size=768),
        optimizer=SimpleNamespace(lr=1e-4, weight_decay=0.01),
        scheduler=SimpleNamespace(warmup_steps=100),
        dataset=SimpleNamespace(data_path="/data/train"),
        logger=SimpleNamespace(log_interval=10),
        tokenizer=SimpleNamespace(tokenizer_model="gpt2"),
        checkpoint=SimpleNamespace(save_interval=1000),
    )

    # Should not raise an error even though _target_ is not in input
    result = build_job_config_from_namespace(ns)

    assert result is not None
    assert result.train.num_steps == 1000


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
            level2=SimpleNamespace(
                level3=SimpleNamespace(level4=SimpleNamespace(value="deep"), other=123)
            )
        )
    )
    result = namespace_to_dict(ns)

    expected = {"level1": {"level2": {"level3": {"level4": {"value": "deep"}, "other": 123}}}}
    assert result == expected
