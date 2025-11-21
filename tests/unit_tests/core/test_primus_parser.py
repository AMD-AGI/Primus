###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for PrimusParser."""

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from primus.core.launcher.parser import PrimusParser


@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a sample YAML config file for testing."""
    config_content = """
work_group: test_group
user_name: test_user
exp_name: test_exp
workspace: /tmp/workspace

modules:
  pretrain_stage:
    module: pre_trainer
    framework: megatron
    config: pre_trainer.yaml
    model: llama_base.yaml

  benchmark_stage:
    module: benchmark_gemm
    op: gemm
    min_size: 128
    max_size: 4096
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture
def cli_args(sample_config_yaml):
    """Create mock CLI arguments."""
    args = argparse.Namespace()
    args.config = sample_config_yaml
    return args


def test_primus_parser_init():
    """Test PrimusParser initialization."""
    parser = PrimusParser()
    assert parser is not None


def test_parse_config(sample_config_yaml):
    """Test parse_config method."""
    parser = PrimusParser()
    parser.parse_config(sample_config_yaml)

    assert parser.config is not None
    assert isinstance(parser.config, SimpleNamespace)
    assert parser.config.config_file == sample_config_yaml
    assert parser.config.work_group == "test_group"
    assert parser.config.user_name == "test_user"
    assert parser.config.exp_name == "test_exp"
    assert parser.config.workspace == "/tmp/workspace"


def test_parse_meta_info(sample_config_yaml):
    """Test parse_meta_info validates required fields."""
    parser = PrimusParser()
    parser.parse_config(sample_config_yaml)

    # Should not raise any exception
    parser.parse_meta_info()


def test_parse_meta_info_missing_field(tmp_path):
    """Test parse_meta_info raises error when required field is missing."""
    config_content = """
work_group: test_group
user_name: test_user
# exp_name is missing
workspace: /tmp/workspace
"""
    config_file = tmp_path / "incomplete_config.yaml"
    config_file.write_text(config_content)

    parser = PrimusParser()
    parser.parse_config(str(config_file))

    with pytest.raises(AssertionError):
        parser.parse_meta_info()


def test_parse_modules(sample_config_yaml):
    """Test parse_modules method."""
    parser = PrimusParser()
    parser.parse_config(sample_config_yaml)
    parser.parse_modules()

    assert hasattr(parser.config, "modules")
    assert hasattr(parser.config.modules, "pretrain_stage")
    assert hasattr(parser.config.modules, "benchmark_stage")

    # Check pretrain_stage module
    pretrain = parser.config.modules.pretrain_stage
    assert pretrain.name == "pretrain_stage"
    assert pretrain.module == "pre_trainer"
    assert pretrain.framework == "megatron"

    # Check benchmark_stage module
    benchmark = parser.config.modules.benchmark_stage
    assert benchmark.name == "benchmark_stage"
    assert benchmark.module == "benchmark_gemm"


def test_parse_modules_missing_module_field(tmp_path):
    """Test parse_modules raises error when 'module' field is missing."""
    config_content = """
work_group: test_group
user_name: test_user
exp_name: test_exp
workspace: /tmp/workspace

modules:
  invalid_stage:
    # module field is missing
    framework: megatron
"""
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text(config_content)

    parser = PrimusParser()
    parser.parse_config(str(config_file))

    with pytest.raises(ValueError, match="missing required 'module' field"):
        parser.parse_modules()


def test_parse_modules_invalid_type(tmp_path):
    """Test parse_modules raises error when module is not a SimpleNamespace."""
    config_content = """
work_group: test_group
user_name: test_user
exp_name: test_exp
workspace: /tmp/workspace

modules:
  invalid_stage: "string_instead_of_dict"
"""
    config_file = tmp_path / "invalid_type_config.yaml"
    config_file.write_text(config_content)

    parser = PrimusParser()
    parser.parse_config(str(config_file))

    with pytest.raises(TypeError, match="must be a SimpleNamespace"):
        parser.parse_modules()


@patch("primus.core.launcher.parser.PrimusConfig")
def test_parse_full_workflow(mock_primus_config, cli_args):
    """Test full parse workflow."""
    parser = PrimusParser()
    parser.parse(cli_args)

    # Verify PrimusConfig was called with correct arguments
    mock_primus_config.assert_called_once()
    call_args = mock_primus_config.call_args
    assert call_args[0][0] == cli_args
    assert isinstance(call_args[0][1], SimpleNamespace)


def test_export(sample_config_yaml, tmp_path):
    """Test export method."""
    parser = PrimusParser()
    parser.parse_config(sample_config_yaml)
    parser.parse_meta_info()
    parser.parse_modules()

    export_path = tmp_path / "exported_config.yaml"
    result_path = parser.export(str(export_path))

    assert result_path.exists()
    assert result_path == export_path


def test_parse_modules_auto_sets_name(tmp_path):
    """Test that parse_modules automatically sets 'name' field if missing."""
    config_content = """
work_group: test_group
user_name: test_user
exp_name: test_exp
workspace: /tmp/workspace

modules:
  my_module:
    module: pre_trainer
    framework: megatron
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    parser = PrimusParser()
    parser.parse_config(str(config_file))
    parser.parse_modules()

    assert parser.config.modules.my_module.name == "my_module"
