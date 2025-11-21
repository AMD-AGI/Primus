###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for YAML utilities."""

import os

import pytest

from primus.core.utils import yaml_utils


@pytest.fixture
def temp_yaml_dir(tmp_path):
    """Create a temporary directory for YAML test files."""
    return tmp_path


def test_parse_yaml_simple(temp_yaml_dir):
    """Test parsing a simple YAML file without includes."""
    yaml_file = temp_yaml_dir / "simple.yaml"
    yaml_file.write_text(
        """
param1: value1
param2: 123
param3:
  nested1: nested_value1
  nested2: nested_value2
"""
    )

    config = yaml_utils.parse_yaml(str(yaml_file))

    assert config["param1"] == "value1"
    assert config["param2"] == 123
    assert config["param3"]["nested1"] == "nested_value1"
    assert config["param3"]["nested2"] == "nested_value2"


def test_parse_yaml_single_include(temp_yaml_dir):
    """Test parsing YAML with a single include."""
    # Create base file
    base_file = temp_yaml_dir / "base.yaml"
    base_file.write_text(
        """
param1: base_value1
param2: base_value2
param3: base_value3
"""
    )

    # Create file that includes base
    main_file = temp_yaml_dir / "main.yaml"
    main_file.write_text(
        """
includes:
  - base.yaml

param2: override_value2
param4: new_value4
"""
    )

    config = yaml_utils.parse_yaml(str(main_file))

    assert config["param1"] == "base_value1"  # From base
    assert config["param2"] == "override_value2"  # Overridden
    assert config["param3"] == "base_value3"  # From base
    assert config["param4"] == "new_value4"  # New parameter


def test_parse_yaml_multiple_includes(temp_yaml_dir):
    """Test parsing YAML with multiple includes (later overrides earlier)."""
    # Create first base file
    base1_file = temp_yaml_dir / "base1.yaml"
    base1_file.write_text(
        """
param1: base1_value1
param2: base1_value2
"""
    )

    # Create second base file
    base2_file = temp_yaml_dir / "base2.yaml"
    base2_file.write_text(
        """
param2: base2_value2
param3: base2_value3
"""
    )

    # Create main file that includes both
    main_file = temp_yaml_dir / "main.yaml"
    main_file.write_text(
        """
includes:
  - base1.yaml
  - base2.yaml

param3: main_value3
"""
    )

    config = yaml_utils.parse_yaml(str(main_file))

    assert config["param1"] == "base1_value1"  # From base1
    assert config["param2"] == "base2_value2"  # base2 overrides base1
    assert config["param3"] == "main_value3"  # main overrides base2


def test_parse_yaml_nested_includes(temp_yaml_dir):
    """Test parsing YAML with nested includes (multi-level)."""
    # Create level 1 base
    level1_file = temp_yaml_dir / "level1.yaml"
    level1_file.write_text(
        """
param1: level1_value1
param2: level1_value2
"""
    )

    # Create level 2 that includes level 1
    level2_file = temp_yaml_dir / "level2.yaml"
    level2_file.write_text(
        """
includes:
  - level1.yaml

param2: level2_value2
param3: level2_value3
"""
    )

    # Create main file that includes level 2
    main_file = temp_yaml_dir / "main.yaml"
    main_file.write_text(
        """
includes:
  - level2.yaml

param3: main_value3
param4: main_value4
"""
    )

    config = yaml_utils.parse_yaml(str(main_file))

    assert config["param1"] == "level1_value1"  # From level1
    assert config["param2"] == "level2_value2"  # level2 overrides level1
    assert config["param3"] == "main_value3"  # main overrides level2
    assert config["param4"] == "main_value4"  # New in main


def test_parse_yaml_deep_merge_nested_dicts(temp_yaml_dir):
    """Test deep merging of nested dictionaries."""
    # Create base file with nested config
    base_file = temp_yaml_dir / "base.yaml"
    base_file.write_text(
        """
optimizer:
  type: adam
  lr: 1e-4
  betas:
    beta1: 0.9
    beta2: 0.999
training:
  epochs: 100
"""
    )

    # Create main file that partially overrides nested config
    main_file = temp_yaml_dir / "main.yaml"
    main_file.write_text(
        """
includes:
  - base.yaml

optimizer:
  lr: 2e-5
  betas:
    beta1: 0.95
training:
  epochs: 200
  batch_size: 32
"""
    )

    config = yaml_utils.parse_yaml(str(main_file))

    assert config["optimizer"]["type"] == "adam"  # Preserved from base
    assert config["optimizer"]["lr"] == 2e-5  # Overridden
    assert config["optimizer"]["betas"]["beta1"] == 0.95  # Overridden
    assert config["optimizer"]["betas"]["beta2"] == 0.999  # Preserved from base
    assert config["training"]["epochs"] == 200  # Overridden
    assert config["training"]["batch_size"] == 32  # New parameter


def test_parse_yaml_env_variable_substitution(temp_yaml_dir):
    """Test environment variable substitution in YAML."""
    os.environ["TEST_VAR"] = "test_value"
    os.environ["TEST_NUM"] = "42"

    yaml_file = temp_yaml_dir / "env.yaml"
    yaml_file.write_text(
        """
param1: ${TEST_VAR}
param2: ${TEST_NUM}
param3: ${UNDEFINED_VAR:default_value}
"""
    )

    config = yaml_utils.parse_yaml(str(yaml_file))

    assert config["param1"] == "test_value"
    assert config["param2"] == 42  # Should be converted to int
    assert config["param3"] == "default_value"

    # Cleanup
    del os.environ["TEST_VAR"]
    del os.environ["TEST_NUM"]


def test_parse_yaml_env_variable_required_missing(temp_yaml_dir):
    """Test that missing required environment variable raises error."""
    yaml_file = temp_yaml_dir / "env_required.yaml"
    yaml_file.write_text(
        """
param1: ${REQUIRED_VAR_MISSING}
"""
    )

    with pytest.raises(ValueError, match="Environment variable 'REQUIRED_VAR_MISSING' is required"):
        yaml_utils.parse_yaml(str(yaml_file))


def test_parse_yaml_empty_file(temp_yaml_dir):
    """Test parsing an empty YAML file."""
    yaml_file = temp_yaml_dir / "empty.yaml"
    yaml_file.write_text("")

    config = yaml_utils.parse_yaml(str(yaml_file))

    assert config == {}


def test_parse_yaml_includes_with_env_vars(temp_yaml_dir):
    """Test includes combined with environment variable substitution."""
    os.environ["TEST_OVERRIDE"] = "env_value"

    base_file = temp_yaml_dir / "base.yaml"
    base_file.write_text(
        """
param1: base_value
param2: base_value2
"""
    )

    main_file = temp_yaml_dir / "main.yaml"
    main_file.write_text(
        """
includes:
  - base.yaml

param2: ${TEST_OVERRIDE}
"""
    )

    config = yaml_utils.parse_yaml(str(main_file))

    assert config["param1"] == "base_value"
    assert config["param2"] == "env_value"

    # Cleanup
    del os.environ["TEST_OVERRIDE"]


def test_parse_yaml_to_namespace(temp_yaml_dir):
    """Test parsing YAML to SimpleNamespace."""
    yaml_file = temp_yaml_dir / "namespace.yaml"
    yaml_file.write_text(
        """
param1: value1
param2:
  nested1: nested_value1
  nested2: nested_value2
"""
    )

    namespace = yaml_utils.parse_yaml_to_namespace(str(yaml_file))

    assert namespace.param1 == "value1"
    assert namespace.param2.nested1 == "nested_value1"
    assert namespace.param2.nested2 == "nested_value2"


def test_deep_merge_dict():
    """Test _deep_merge_dict helper function."""
    target = {"a": 1, "b": {"b1": 2, "b2": 3}, "c": 4}

    source = {"b": {"b2": 30, "b3": 40}, "d": 5}

    yaml_utils._deep_merge_dict(target, source)

    assert target["a"] == 1  # Unchanged
    assert target["b"]["b1"] == 2  # Unchanged
    assert target["b"]["b2"] == 30  # Overridden
    assert target["b"]["b3"] == 40  # Added
    assert target["c"] == 4  # Unchanged
    assert target["d"] == 5  # Added


def test_parse_yaml_append_new_params(temp_yaml_dir):
    """Test that new parameters can be appended without existing in base."""
    base_file = temp_yaml_dir / "base.yaml"
    base_file.write_text(
        """
existing_param: value1
"""
    )

    main_file = temp_yaml_dir / "main.yaml"
    main_file.write_text(
        """
includes:
  - base.yaml

existing_param: override_value
new_param1: new_value1
new_param2: new_value2
completely_new_nested:
  nested1: value1
  nested2: value2
"""
    )

    config = yaml_utils.parse_yaml(str(main_file))

    assert config["existing_param"] == "override_value"
    assert config["new_param1"] == "new_value1"
    assert config["new_param2"] == "new_value2"
    assert config["completely_new_nested"]["nested1"] == "value1"
    assert config["completely_new_nested"]["nested2"] == "value2"


def test_parse_yaml_complex_multi_level_scenario(temp_yaml_dir):
    """Test a complex real-world scenario with multiple levels of includes."""
    # Level 1: module_base.yaml
    module_base = temp_yaml_dir / "module_base.yaml"
    module_base.write_text(
        """
log_interval: 1
save_interval: 1000
checkpoint_dir: ./checkpoints
"""
    )

    # Level 2: primus_megatron_module.yaml
    primus_megatron = temp_yaml_dir / "primus_megatron_module.yaml"
    primus_megatron.write_text(
        """
includes:
  - module_base.yaml

framework: megatron
use_flash_attn: true
optimizer:
  type: adam
  lr: 1e-4
"""
    )

    # Level 3: primus_turbo.yaml
    primus_turbo = temp_yaml_dir / "primus_turbo.yaml"
    primus_turbo.write_text(
        """
use_turbo: true
turbo_config:
  enable_overlap: true
"""
    )

    # Level 4: pre_trainer.yaml
    pre_trainer = temp_yaml_dir / "pre_trainer.yaml"
    pre_trainer.write_text(
        """
includes:
  - primus_megatron_module.yaml
  - primus_turbo.yaml

trainable: true
train_iters: 1000
optimizer:
  lr: 2e-5
  beta1: 0.9
experimental_feature: true
"""
    )

    config = yaml_utils.parse_yaml(str(pre_trainer))

    # Check values from all levels
    assert config["log_interval"] == 1  # From module_base
    assert config["save_interval"] == 1000  # From module_base
    assert config["framework"] == "megatron"  # From primus_megatron
    assert config["use_flash_attn"] is True  # From primus_megatron
    assert config["use_turbo"] is True  # From primus_turbo
    assert config["turbo_config"]["enable_overlap"] is True  # From primus_turbo
    assert config["trainable"] is True  # From pre_trainer
    assert config["train_iters"] == 1000  # From pre_trainer
    assert config["optimizer"]["type"] == "adam"  # From primus_megatron
    assert config["optimizer"]["lr"] == 2e-5  # Overridden in pre_trainer
    assert config["optimizer"]["beta1"] == 0.9  # New in pre_trainer
    assert config["experimental_feature"] is True  # New in pre_trainer
