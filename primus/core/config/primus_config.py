###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
PrimusConfig: Unified configuration loader for Primus workflows.

Responsibilities:
    - Load and parse YAML configuration files with extends support
    - Validate workspace metadata (work_group, user_name, exp_name)
    - Load platform configuration with overrides
    - Parse module list and load model presets if specified
    - Merge model presets with user params (user params have higher priority)

The YAML structure expected:
    work_group: amd
    user_name: xiaoming
    exp_name: exp_pretrain
    workspace: ./output

    modules:
      - name: pretrain_stage
        module: pre_trainer
        framework: megatron
        model: llama2_7B
        params:
          batch_size: 32
"""

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from primus.core.config import yaml_loader
from primus.core.config.merge_utils import deep_merge
from primus.core.config.model_preset_loader import ModelPresetLoader
from primus.core.utils import file_utils, yaml_utils


@dataclass
class ModuleConfig:
    """
    Parsed per-module configuration in Primus.
    """

    name: str
    module: str  # e.g., pretrain, benchmark_gemm, sft
    framework: Optional[str] = None  # e.g., megatron, torchtitan (optional)
    model: Optional[str] = None  # e.g., llama2_7B (optional, triggers preset loading)
    params: Dict[str, Any] = field(default_factory=dict)  # merged params (preset + overrides)


class PrimusConfig:
    """
    Primus configuration container with model preset support.

    After loading, modules are accessible via:
        config.get_module_config("pre_trainer")  # Returns ModuleConfig
    """

    def __init__(
        self,
        cli_args: argparse.Namespace,
        work_group: str,
        user_name: str,
        exp_name: str,
        workspace: str,
        platform: SimpleNamespace,
        modules: List[ModuleConfig],
    ):
        """
        Initialize PrimusConfig.

        Args:
            cli_args: Command-line arguments
            work_group: Work group identifier
            user_name: User name
            exp_name: Experiment name
            workspace: Workspace directory path
            platform: Platform configuration namespace
            modules: List of parsed module configurations
        """
        self._cli_args = cli_args
        self._work_group = work_group
        self._user_name = user_name
        self._exp_name = exp_name
        self._workspace = workspace
        self._platform = platform
        self._modules = {m.name: m for m in modules}

        # Create experiment root directory
        self._exp_root_path = os.path.join(workspace, work_group, user_name, exp_name)
        file_utils.create_path_if_not_exists(self._exp_root_path)

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

        # Load YAML → dict (handles extends, env vars)
        config_dict = yaml_loader.parse_yaml(config_file)

        # Validate required metadata
        cls._validate_meta_info(config_dict)

        # Load platform configuration
        platform = cls._load_platform(config_dict, primus_home)

        # Parse modules with model preset loading
        modules = cls._parse_modules(config_dict.get("modules", []))

        return cls(
            cli_args=cli_args,
            work_group=config_dict["work_group"],
            user_name=config_dict["user_name"],
            exp_name=config_dict["exp_name"],
            workspace=config_dict["workspace"],
            platform=platform,
            modules=modules,
        )

    @staticmethod
    def _validate_meta_info(config_dict: dict):
        """Validate required workspace metadata fields."""
        required = ["work_group", "user_name", "exp_name", "workspace"]
        missing = [key for key in required if key not in config_dict]
        if missing:
            raise ValueError(f"Configuration missing required fields: {', '.join(missing)}")

    @staticmethod
    def _load_platform(config_dict: dict, primus_home: Path) -> SimpleNamespace:
        """Load platform configuration with defaults and overrides."""
        # Set default platform if not specified
        platform_config = config_dict.get("platform", {})
        if not platform_config:
            platform_config = {
                "config": "platform_azure.yaml",
                "overrides": {"master_sink_level": "INFO"},
            }

        # Load platform YAML
        platform_path = (
            primus_home / "configs" / "platforms" / platform_config.get("config", "platform_azure.yaml")
        )
        platform_cfg = yaml_utils.parse_yaml_to_namespace(str(platform_path))
        platform_cfg.config = platform_config.get("config", "platform_azure.yaml")

        # Apply user overrides
        if "overrides" in platform_config:
            overrides_ns = (
                SimpleNamespace(**platform_config["overrides"])
                if isinstance(platform_config["overrides"], dict)
                else platform_config["overrides"]
            )
            yaml_utils.override_namespace(platform_cfg, overrides_ns)

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

        return platform_cfg

    @staticmethod
    def _parse_modules(modules_list: list) -> List[ModuleConfig]:
        """
        Parse modules list with model preset loading.

        For each module:
            1. If 'model' is specified, load model preset via ModelPresetLoader
            2. Merge preset with user params (user params override preset)
            3. Create ModuleConfig instance
        """
        parsed_modules = []

        for idx, module_dict in enumerate(modules_list):
            if not isinstance(module_dict, dict):
                raise ValueError(f"Module at index {idx} must be a dict, got {type(module_dict)}")

            # Extract module fields
            name = module_dict.get("name")
            module_type = module_dict.get("module")
            framework = module_dict.get("framework")
            model_name = module_dict.get("model")
            user_params = module_dict.get("params", {})

            # Validate required fields
            if not module_type:
                raise ValueError(f"Module at index {idx} missing required 'module' field")

            if not name:
                name = f"{module_type}_{idx}"

            # Load model preset if specified
            merged_params = {}
            if model_name:
                if not framework:
                    raise ValueError(
                        f"Module '{name}' uses model preset '{model_name}' "
                        f"but has no 'framework' specified."
                    )

                # Load model preset
                model_preset = ModelPresetLoader.load(model_name, framework)

                # Merge: model preset (base) + user params (override)
                merged_params = deep_merge(model_preset, user_params)
            else:
                merged_params = user_params

            # Create ModuleConfig
            parsed_modules.append(
                ModuleConfig(
                    name=name,
                    module=module_type,
                    framework=framework,
                    model=model_name,
                    params=merged_params,
                )
            )

        return parsed_modules

    # ================================================================
    # Properties
    # ================================================================

    @property
    def cli_args(self) -> argparse.Namespace:
        """Command-line arguments."""
        return self._cli_args

    @property
    def exp_root_path(self) -> str:
        """Root directory for this experiment's outputs (configs, logs, checkpoints)."""
        return self._exp_root_path

    @property
    def config_meta_info(self) -> dict:
        """Workspace metadata (work_group, user_name, exp_name)."""
        return {
            "work_group": self._work_group,
            "user_name": self._user_name,
            "exp_name": self._exp_name,
        }

    @property
    def platform_config(self) -> SimpleNamespace:
        """Platform configuration."""
        return self._platform

    @property
    def module_keys(self) -> list:
        """List of module names."""
        return list(self._modules.keys())

    # ================================================================
    # Module API
    # ================================================================

    def get_module_config(self, module_type: str) -> ModuleConfig:
        """
        Get configuration for a module by its type.

        Args:
            module_type: Type of the module (e.g., "pre_trainer", "benchmark_gemm")

        Returns:
            ModuleConfig instance containing module configuration

        Raises:
            ValueError: If no module with the specified type is found

        Example:
            # YAML has: - name: pretrain_stage
            #             module: pre_trainer
            config.get_module_config("pre_trainer")  # Returns pretrain_stage ModuleConfig
        """
        # Find module by type
        for module in self._modules.values():
            if module.module == module_type:
                return module

        # Not found - list available module types
        available_types = [f"{m.module} (name: {m.name})" for m in self._modules.values()]
        raise ValueError(
            f"No module with type '{module_type}' found. " f"Available modules: {', '.join(available_types)}"
        )

    def get_module_by_name(self, name: str) -> ModuleConfig:
        """
        Get configuration for a module by its name.

        Args:
            name: Name of the module

        Returns:
            ModuleConfig instance

        Raises:
            ValueError: If module name not found
        """
        if name not in self._modules:
            raise ValueError(
                f"Module '{name}' not found. Available modules: {', '.join(self._modules.keys())}"
            )
        return self._modules[name]

    def module_config_path(self, module_name: str) -> str:
        """Get export path for module configuration."""
        return os.path.join(self._exp_root_path, f"{module_name}.yaml")

    def export_module_config(self, module_name: str) -> str:
        """
        Export module configuration to YAML file.

        Args:
            module_name: Name of module to export

        Returns:
            Path to exported YAML file
        """
        module_config = self.get_module_by_name(module_name)
        config_path = self.module_config_path(module_name)

        # Convert ModuleConfig to dict for export
        config_dict = {
            "name": module_config.name,
            "module": module_config.module,
            "framework": module_config.framework,
            "model": module_config.model,
            "params": module_config.params,
        }
        yaml_utils.dump_namespace_to_yaml(config_dict, config_path)
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

        data = {
            "work_group": self._work_group,
            "user_name": self._user_name,
            "exp_name": self._exp_name,
            "workspace": self._workspace,
            "platform": yaml_utils.nested_namespace_to_dict(self._platform),
            "modules": [
                {
                    "name": m.name,
                    "module": m.module,
                    "framework": m.framework,
                    "model": m.model,
                    "params": m.params,
                }
                for m in self._modules.values()
            ],
        }
        yaml_utils.dump_namespace_to_yaml(data, str(path))
        print(f"[PrimusConfig] Exported merged config to {path}")
        return path

    def __str__(self) -> str:
        """Human-readable representation of configuration."""
        lines = [
            f"PrimusConfig:",
            f"  work_group: {self._work_group}",
            f"  user_name: {self._user_name}",
            f"  exp_name: {self._exp_name}",
            f"  workspace: {self._workspace}",
            f"  modules: {len(self._modules)}",
        ]
        for module in self._modules.values():
            lines.append(f"    - {module.name} ({module.module})")
        return "\n".join(lines)
