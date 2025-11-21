###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for PrimusConfig and ModuleConfig.
"""

import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from primus.core.config.primus_config import ModuleConfig, PrimusConfig


class TestModuleConfig:
    """Test ModuleConfig dataclass."""

    def test_module_config_creation_minimal(self):
        """Test creating ModuleConfig with minimal required fields."""
        config = ModuleConfig(name="test_module", module="pre_trainer")

        assert config.name == "test_module"
        assert config.module == "pre_trainer"
        assert config.framework is None
        assert config.model is None
        assert config.params == {}

    def test_module_config_creation_full(self):
        """Test creating ModuleConfig with all fields."""
        params = {"batch_size": 32, "lr": 1e-4}
        config = ModuleConfig(
            name="pretrain_stage",
            module="pre_trainer",
            framework="megatron",
            model="llama2_7B",
            params=params,
        )

        assert config.name == "pretrain_stage"
        assert config.module == "pre_trainer"
        assert config.framework == "megatron"
        assert config.model == "llama2_7B"
        assert config.params == params

    def test_module_config_default_params(self):
        """Test that params defaults to empty dict."""
        config = ModuleConfig(name="test", module="benchmark")
        assert isinstance(config.params, dict)
        assert len(config.params) == 0


class TestPrimusConfigValidation:
    """Test PrimusConfig validation methods."""

    def test_validate_meta_info_success(self):
        """Test successful metadata validation."""
        config_dict = {
            "work_group": "amd",
            "user_name": "xiaoming",
            "exp_name": "test_exp",
            "workspace": "/tmp/workspace",
        }
        # Should not raise
        PrimusConfig._validate_meta_info(config_dict)

    def test_validate_meta_info_missing_field(self):
        """Test validation fails when required field is missing."""
        config_dict = {
            "work_group": "amd",
            "user_name": "xiaoming",
            # Missing exp_name and workspace
        }
        with pytest.raises(ValueError, match="Configuration missing required fields"):
            PrimusConfig._validate_meta_info(config_dict)

    def test_validate_meta_info_empty_dict(self):
        """Test validation fails with empty config."""
        with pytest.raises(ValueError, match="Configuration missing required fields"):
            PrimusConfig._validate_meta_info({})


class TestPrimusConfigParseModules:
    """Test module parsing logic."""

    def test_parse_modules_minimal(self):
        """Test parsing module with minimal fields."""
        modules_list = [{"module": "pre_trainer"}]

        modules = PrimusConfig._parse_modules(modules_list)

        assert len(modules) == 1
        assert modules[0].name == "pre_trainer_0"  # Auto-generated name
        assert modules[0].module == "pre_trainer"
        assert modules[0].framework is None
        assert modules[0].model is None
        assert modules[0].params == {}

    def test_parse_modules_with_name(self):
        """Test parsing module with explicit name."""
        modules_list = [{"name": "my_trainer", "module": "pre_trainer"}]

        modules = PrimusConfig._parse_modules(modules_list)

        assert len(modules) == 1
        assert modules[0].name == "my_trainer"
        assert modules[0].module == "pre_trainer"

    def test_parse_modules_with_params(self):
        """Test parsing module with params."""
        modules_list = [
            {
                "name": "trainer",
                "module": "pre_trainer",
                "framework": "megatron",
                "params": {"batch_size": 32, "lr": 1e-4},
            }
        ]

        modules = PrimusConfig._parse_modules(modules_list)

        assert len(modules) == 1
        assert modules[0].params == {"batch_size": 32, "lr": 1e-4}

    def test_parse_modules_missing_module_field(self):
        """Test parsing fails when module field is missing."""
        modules_list = [{"name": "test"}]  # Missing 'module' field

        with pytest.raises(ValueError, match="missing required 'module' field"):
            PrimusConfig._parse_modules(modules_list)

    def test_parse_modules_invalid_type(self):
        """Test parsing fails when module is not a dict."""
        modules_list = ["invalid_module"]

        with pytest.raises(ValueError, match="must be a dict"):
            PrimusConfig._parse_modules(modules_list)

    def test_parse_modules_model_without_framework(self):
        """Test parsing fails when model is specified without framework."""
        modules_list = [
            {
                "name": "trainer",
                "module": "pre_trainer",
                "model": "llama2_7B",  # Has model
                # Missing framework
            }
        ]

        with pytest.raises(ValueError, match="but has no 'framework' specified"):
            PrimusConfig._parse_modules(modules_list)

    def test_parse_modules_multiple(self):
        """Test parsing multiple modules."""
        modules_list = [
            {"name": "trainer1", "module": "pre_trainer"},
            {"name": "trainer2", "module": "sft_trainer"},
            {"module": "benchmark_gemm"},  # No name
        ]

        modules = PrimusConfig._parse_modules(modules_list)

        assert len(modules) == 3
        assert modules[0].name == "trainer1"
        assert modules[1].name == "trainer2"
        assert modules[2].name == "benchmark_gemm_2"  # Auto-generated


class TestPrimusConfigFromFile:
    """Test PrimusConfig.from_file() method."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for test configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def minimal_config_file(self, temp_config_dir):
        """Create a minimal valid config file."""
        config_path = temp_config_dir / "config.yaml"
        config_data = {
            "work_group": "test_group",
            "user_name": "test_user",
            "exp_name": "test_exp",
            "workspace": str(temp_config_dir / "workspace"),
            "modules": [{"name": "trainer", "module": "pre_trainer"}],
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return config_path

    def test_from_file_minimal(self, minimal_config_file):
        """Test loading minimal config from file."""
        cli_args = argparse.Namespace()

        config = PrimusConfig.from_file(str(minimal_config_file), cli_args)

        assert config._work_group == "test_group"
        assert config._user_name == "test_user"
        assert config._exp_name == "test_exp"
        assert len(config._modules) == 1
        assert "trainer" in config._modules

    def test_from_file_with_extends(self, temp_config_dir):
        """Test loading config with extends."""
        # Create base config
        base_config_path = temp_config_dir / "base.yaml"
        base_data = {"batch_size": 16, "lr": 1e-3}
        with open(base_config_path, "w") as f:
            yaml.dump(base_data, f)

        # Create main config that extends base
        main_config_path = temp_config_dir / "main.yaml"
        main_data = {
            "extends": ["base.yaml"],
            "work_group": "test_group",
            "user_name": "test_user",
            "exp_name": "test_exp",
            "workspace": str(temp_config_dir / "workspace"),
            "batch_size": 32,  # Override base
            "modules": [{"name": "trainer", "module": "pre_trainer"}],
        }
        with open(main_config_path, "w") as f:
            yaml.dump(main_data, f)

        cli_args = argparse.Namespace()
        config = PrimusConfig.from_file(str(main_config_path), cli_args)

        assert config._work_group == "test_group"

    def test_from_file_missing_required_fields(self, temp_config_dir):
        """Test loading config with missing required fields."""
        config_path = temp_config_dir / "invalid.yaml"
        config_data = {
            "work_group": "test_group",
            # Missing user_name, exp_name, workspace
            "modules": [],
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cli_args = argparse.Namespace()
        with pytest.raises(ValueError, match="Configuration missing required fields"):
            PrimusConfig.from_file(str(config_path), cli_args)


class TestPrimusConfigProperties:
    """Test PrimusConfig properties."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample PrimusConfig instance."""
        cli_args = argparse.Namespace(config="test.yaml")
        platform = SimpleNamespace(name="azure", master_sink_level="INFO")
        modules = [
            ModuleConfig(name="trainer1", module="pre_trainer"),
            ModuleConfig(name="trainer2", module="sft_trainer"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PrimusConfig(
                cli_args=cli_args,
                work_group="test_group",
                user_name="test_user",
                exp_name="test_exp",
                workspace=tmpdir,
                platform=platform,
                modules=modules,
            )
            yield config

    def test_cli_args_property(self, sample_config):
        """Test cli_args property."""
        assert sample_config.cli_args.config == "test.yaml"

    def test_config_root_path_property(self, sample_config):
        """Test config_root_path property."""
        assert "test_group" in sample_config.config_root_path
        assert "test_user" in sample_config.config_root_path
        assert "test_exp" in sample_config.config_root_path

    def test_config_meta_info_property(self, sample_config):
        """Test config_meta_info property."""
        meta = sample_config.config_meta_info
        assert meta["work_group"] == "test_group"
        assert meta["user_name"] == "test_user"
        assert meta["exp_name"] == "test_exp"

    def test_platform_config_property(self, sample_config):
        """Test platform_config property."""
        platform = sample_config.platform_config
        assert platform.name == "azure"
        assert platform.master_sink_level == "INFO"

    def test_module_keys_property(self, sample_config):
        """Test module_keys property."""
        keys = sample_config.module_keys
        assert len(keys) == 2
        assert "trainer1" in keys
        assert "trainer2" in keys


class TestPrimusConfigModuleAPI:
    """Test PrimusConfig module access methods."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample PrimusConfig with multiple modules."""
        cli_args = argparse.Namespace()
        platform = SimpleNamespace(name="azure")
        modules = [
            ModuleConfig(name="pretrain_stage", module="pre_trainer", framework="megatron"),
            ModuleConfig(name="sft_stage", module="sft_trainer", framework="megatron"),
            ModuleConfig(name="benchmark_task", module="benchmark_gemm"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PrimusConfig(
                cli_args=cli_args,
                work_group="test",
                user_name="user",
                exp_name="exp",
                workspace=tmpdir,
                platform=platform,
                modules=modules,
            )
            yield config

    def test_get_module_config_by_type(self, sample_config):
        """Test getting module by type."""
        module = sample_config.get_module_config("pre_trainer")

        assert module.name == "pretrain_stage"
        assert module.module == "pre_trainer"
        assert module.framework == "megatron"

    def test_get_module_config_not_found(self, sample_config):
        """Test getting non-existent module by type."""
        with pytest.raises(ValueError, match="No module with type 'non_existent' found"):
            sample_config.get_module_config("non_existent")

    def test_get_module_by_name(self, sample_config):
        """Test getting module by name."""
        module = sample_config.get_module_by_name("sft_stage")

        assert module.name == "sft_stage"
        assert module.module == "sft_trainer"

    def test_get_module_by_name_not_found(self, sample_config):
        """Test getting non-existent module by name."""
        with pytest.raises(ValueError, match="Module 'invalid_name' not found"):
            sample_config.get_module_by_name("invalid_name")

    def test_module_config_path(self, sample_config):
        """Test getting module config export path."""
        path = sample_config.module_config_path("pretrain_stage")

        assert "pretrain_stage.yaml" in path
        assert sample_config.config_root_path in path


class TestPrimusConfigExport:
    """Test PrimusConfig export functionality."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample PrimusConfig."""
        cli_args = argparse.Namespace()
        platform = SimpleNamespace(name="azure", master_sink_level="INFO")
        modules = [
            ModuleConfig(
                name="trainer",
                module="pre_trainer",
                framework="megatron",
                model="llama2_7B",
                params={"batch_size": 32},
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PrimusConfig(
                cli_args=cli_args,
                work_group="test",
                user_name="user",
                exp_name="exp",
                workspace=tmpdir,
                platform=platform,
                modules=modules,
            )
            yield config

    def test_export_full_config(self, sample_config, tmp_path):
        """Test exporting full configuration."""
        export_path = tmp_path / "exported_config.yaml"

        result_path = sample_config.export(str(export_path))

        assert result_path.exists()
        with open(result_path) as f:
            exported_data = yaml.safe_load(f)

        assert exported_data["work_group"] == "test"
        assert exported_data["user_name"] == "user"
        assert exported_data["exp_name"] == "exp"
        assert len(exported_data["modules"]) == 1
        assert exported_data["modules"][0]["name"] == "trainer"

    def test_str_representation(self, sample_config):
        """Test string representation of PrimusConfig."""
        str_repr = str(sample_config)

        assert "PrimusConfig:" in str_repr
        assert "work_group: test" in str_repr
        assert "user_name: user" in str_repr
        assert "exp_name: exp" in str_repr
        assert "trainer (pre_trainer)" in str_repr


class TestPrimusConfigIntegration:
    """Integration tests for PrimusConfig."""

    @pytest.fixture
    def complex_config_file(self, tmp_path):
        """Create a complex config file for integration testing."""
        config_path = tmp_path / "complex_config.yaml"
        config_data = {
            "work_group": "amd",
            "user_name": "xiaoming",
            "exp_name": "llama_pretrain",
            "workspace": str(tmp_path / "workspace"),
            "modules": [
                {
                    "name": "pretrain_stage",
                    "module": "pre_trainer",
                    "framework": "megatron",
                    "params": {"batch_size": 32, "lr": 1e-4, "num_layers": 32},
                },
                {
                    "name": "benchmark_task",
                    "module": "benchmark_gemm",
                    "params": {"dtype": "bf16", "size": 4096},
                },
            ],
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return config_path

    def test_full_workflow(self, complex_config_file):
        """Test full workflow: load, access, export."""
        cli_args = argparse.Namespace(config=str(complex_config_file))

        # Load config
        config = PrimusConfig.from_file(str(complex_config_file), cli_args)

        # Access modules
        pretrain_module = config.get_module_config("pre_trainer")
        assert pretrain_module.name == "pretrain_stage"
        assert pretrain_module.params["batch_size"] == 32

        benchmark_module = config.get_module_by_name("benchmark_task")
        assert benchmark_module.module == "benchmark_gemm"
        assert benchmark_module.params["dtype"] == "bf16"

        # Check properties
        assert config.config_meta_info["work_group"] == "amd"
        assert len(config.module_keys) == 2

        # Export
        export_path = Path(complex_config_file).parent / "exported.yaml"
        config.export(str(export_path))
        assert export_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
