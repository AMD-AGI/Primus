###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus configuration parser."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from primus.core.launcher.config import PrimusConfig
from primus.core.utils import constant_vars, yaml_utils


class PrimusParser(object):
    def __init__(self):
        pass

    def parse(self, cli_args: argparse.Namespace) -> PrimusConfig:
        config_file = cli_args.config
        self.primus_home = Path(os.path.dirname(__file__)).parent.parent.absolute()
        self.parse_config(config_file)
        self.parse_meta_info()
        self.parse_platform()
        self.parse_modules()
        return PrimusConfig(cli_args, self.config)

    def parse_config(self, config_file: str):
        self.config = yaml_utils.parse_yaml_to_namespace(config_file)
        self.config.name = constant_vars.PRIMUS_CONFIG_NAME
        self.config.config_file = config_file

    def parse_meta_info(self):
        yaml_utils.check_key_in_namespace(self.config, "work_group")
        yaml_utils.check_key_in_namespace(self.config, "user_name")
        yaml_utils.check_key_in_namespace(self.config, "exp_name")
        yaml_utils.check_key_in_namespace(self.config, "workspace")

    def parse_platform(self):
        # If platform is set in config
        if not hasattr(self.config, "platform"):
            self.config.platform = SimpleNamespace(
                config="platform_azure.yaml", overrides=SimpleNamespace(master_sink_level="INFO")
            )

        # Load platform config
        config_path = os.path.join(self.primus_home, "configs/platforms", self.config.platform.config)
        platform_config = yaml_utils.parse_yaml_to_namespace(config_path)
        platform_config.config = self.config.platform.config

        # Optional overrides
        if yaml_utils.has_key_in_namespace(self.config.platform, "overrides"):
            yaml_utils.override_namespace(platform_config, self.config.platform.overrides)

        # Final required key checks
        for key in [
            "name",
            "num_nodes_env_key",
            "node_rank_env_key",
            "master_addr_env_key",
            "master_port_env_key",
            "gpus_per_node_env_key",
            "master_sink_level",
            "workspace",
        ]:
            yaml_utils.check_key_in_namespace(platform_config, key)

        yaml_utils.set_value_by_key(self.config, "platform", platform_config, allow_override=True)

    def get_model_format(self, framework: str):
        map = {
            "megatron": "megatron",
            "light-megatron": "megatron",
            "torchtitan": "torchtitan",
            "maxtext": "maxtext",
        }
        assert framework in map, f"Invalid module framework: {framework}."
        return map[framework]

    def parse_modules(self):
        """
        Parse modules section from YAML config.
        Each module must have 'name' and 'module' fields.
        """
        yaml_utils.check_key_in_namespace(self.config, "modules")

        # Iterate over normalized modules (SimpleNamespace keyed by name)
        for module_name, module_def in vars(self.config.modules).items():
            if not isinstance(module_def, SimpleNamespace):
                raise TypeError(f"Module '{module_name}' must be a SimpleNamespace, got {type(module_def)}")

            # Ensure 'module' field exists (module type)
            if not hasattr(module_def, "module"):
                raise ValueError(f"Module '{module_name}' is missing required 'module' field")

            # Ensure 'name' field is set
            if not hasattr(module_def, "name"):
                module_def.name = module_name

            # Module is already parsed as SimpleNamespace, no further processing needed
            yaml_utils.set_value_by_key(self.config.modules, module_name, module_def, allow_override=True)

    def export(self, export_path):
        """
        Export the merged Primus config to YAML.
        """
        path = Path(export_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = yaml_utils.nested_namespace_to_dict(self.config)
        yaml_utils.dump_namespace_to_yaml(data, str(path))
        print(f"[PrimusConfig] Exported merged config to {path}")
        return path
