###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
PrimusConfig: Unified configuration loader for Primus workflows.

Responsibilities:
    - Load and parse YAML configuration files
    - Validate workspace metadata (work_group, user_name, exp_name)
    - Load platform configuration with overrides
    - Parse module list into accessible namespace structure

The YAML structure expected:
    work_group: amd
    user_name: xiaoming
    exp_name: exp_pretrain
    workspace: ./output

    modules:
      - name: gemm_bench
        module: benchmark_gemm
        dtype: bf16
      - name: pretrain_stage
        module: pre_trainer
        framework: megatron
        model: llama2_7B
"""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from primus.core.utils import constant_vars, file_utils, yaml_utils


class PrimusConfig:
    """
    Primus configuration container.

    After loading, modules are accessible via:
        config.get_module("pretrain_stage")  # Returns SimpleNamespace
    """

    def __init__(self, cli_args: argparse.Namespace, config: SimpleNamespace):
        """
        Initialize PrimusConfig.

        Args:
            cli_args: Command-line arguments
            config: Parsed YAML configuration as SimpleNamespace
        """
        self._cli_args = cli_args
        self._config = config

        # Create workspace directory structure
        self._config_root_path = os.path.join(
            config.workspace, config.work_group, config.user_name, config.exp_name
        )
        file_utils.create_path_if_not_exists(self._config_root_path)

    @classmethod
    def from_file(cls, config_file: str, cli_args: argparse.Namespace) -> "PrimusConfig":
        """
        Load and parse Primus configuration from YAML file.

        Args:
            config_file: Path to YAML configuration file
            cli_args: Parsed command-line arguments

        Returns:
            PrimusConfig instance with fully loaded configuration

        Raises:
            ValueError: If required fields are missing or invalid
        """
        primus_home = Path(__file__).resolve().parent.parent.parent

        # Load YAML → SimpleNamespace (handles includes, env vars, modules list)
        config = yaml_utils.parse_yaml_to_namespace(config_file)
        config.name = constant_vars.PRIMUS_CONFIG_NAME
        config.config_file = config_file

        # Validate required metadata
        cls._validate_meta_info(config)

        # Load and merge platform configuration
        cls._load_platform(config, primus_home)

        # Validate modules structure (already parsed by yaml_utils)
        cls._validate_modules(config)

        return cls(cli_args, config)

    @staticmethod
    def _validate_meta_info(config: SimpleNamespace):
        """Validate required workspace metadata fields."""
        required = ["work_group", "user_name", "exp_name", "workspace"]
        missing = [key for key in required if not hasattr(config, key)]
        if missing:
            raise ValueError(f"Configuration missing required fields: {', '.join(missing)}")

    @staticmethod
    def _load_platform(config: SimpleNamespace, primus_home: Path):
        """Load platform configuration with defaults and overrides."""
        # Set default platform if not specified
        if not hasattr(config, "platform"):
            config.platform = SimpleNamespace(
                config="platform_azure.yaml",
                overrides=SimpleNamespace(master_sink_level="INFO"),
            )

        # Load platform YAML
        platform_path = primus_home / "configs" / "platforms" / config.platform.config
        platform_cfg = yaml_utils.parse_yaml_to_namespace(str(platform_path))
        platform_cfg.config = config.platform.config

        # Apply user overrides
        if hasattr(config.platform, "overrides"):
            yaml_utils.override_namespace(platform_cfg, config.platform.overrides)

        # Validate required platform keys
        required_keys = [
            "name",
            "num_nodes_env_key",
            "node_rank_env_key",
            "master_addr_env_key",
            "master_port_env_key",
            "gpus_per_node_env_key",
            "master_sink_level",
            "workspace",
        ]
        missing = [key for key in required_keys if not hasattr(platform_cfg, key)]
        if missing:
            raise ValueError(f"Platform config missing required keys: {', '.join(missing)}")

        config.platform = platform_cfg

    @staticmethod
    def _validate_modules(config: SimpleNamespace):
        """
        Validate modules structure.

        After yaml_utils.dict_to_nested_namespace, modules should be a
        SimpleNamespace with module names as attributes.
        """
        if not hasattr(config, "modules"):
            raise ValueError("Configuration must have 'modules' section")

        if not isinstance(config.modules, SimpleNamespace):
            raise ValueError(
                f"'modules' must be a SimpleNamespace (got {type(config.modules)}). "
                "Ensure YAML has 'modules' as a list with 'name' fields."
            )

        # Validate each module has required 'name' and 'module' fields
        for module_name in dir(config.modules):
            if module_name.startswith("_"):
                continue
            module = getattr(config.modules, module_name)
            if not hasattr(module, "name"):
                raise ValueError(f"Module '{module_name}' missing 'name' field")
            if not hasattr(module, "module"):
                raise ValueError(f"Module '{module_name}' missing 'module' field")

    # ================================================================
    # Properties
    # ================================================================

    @property
    def cli_args(self) -> argparse.Namespace:
        """Command-line arguments."""
        return self._cli_args

    @property
    def config_root_path(self) -> str:
        """Root directory for this experiment's outputs."""
        return self._config_root_path

    @property
    def config_meta_info(self) -> dict:
        """Workspace metadata (work_group, user_name, exp_name)."""
        return {
            "work_group": self._config.work_group,
            "user_name": self._config.user_name,
            "exp_name": self._config.exp_name,
        }

    @property
    def platform_config(self) -> SimpleNamespace:
        """Platform configuration."""
        return self._config.platform

    @property
    def module_keys(self) -> list:
        """List of module names."""
        return [k for k in dir(self._config.modules) if not k.startswith("_")]

    # ================================================================
    # Module API
    # ================================================================

    def get_module_config(self, module_name: str) -> SimpleNamespace:
        """
        Get configuration for a specific module.

        Args:
            module_name: Name of the module (e.g., "pretrain_stage")

        Returns:
            SimpleNamespace containing module configuration

        Raises:
            ValueError: If module not found
        """
        if not hasattr(self._config.modules, module_name):
            available = self.module_keys
            raise ValueError(
                f"Module '{module_name}' not found. " f"Available modules: {', '.join(available)}"
            )
        return getattr(self._config.modules, module_name)

    def module_config_path(self, module_name: str) -> str:
        """Get export path for module configuration."""
        return os.path.join(self._config_root_path, f"{module_name}.yaml")

    def export_module_config(self, module_name: str) -> str:
        """
        Export module configuration to YAML file.

        Args:
            module_name: Name of module to export

        Returns:
            Path to exported YAML file
        """
        module_config = self.get_module_config(module_name)
        config_path = self.module_config_path(module_name)
        yaml_utils.dump_namespace_to_yaml(module_config, config_path)
        return config_path

    def export(self, export_path: str) -> Path:
        """
        Export full configuration to YAML file.

        Args:
            export_path: Destination file path

        Returns:
            Resolved path to exported file
        """
        path = Path(export_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        data = yaml_utils.nested_namespace_to_dict(self._config)
        yaml_utils.dump_namespace_to_yaml(data, str(path))
        print(f"[PrimusConfig] Exported merged config to {path}")
        return path

    def __str__(self) -> str:
        """Human-readable representation of configuration."""
        return yaml_utils.parse_nested_namespace_to_str(self._config)
