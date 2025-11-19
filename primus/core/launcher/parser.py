import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from primus.core.launcher.config import PrimusConfig
from primus.core.utils import constant_vars, yaml_utils


def _parse_args(extra_args_provider=None, ignore_unknown_args=False) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Primus Arguments", allow_abbrev=False)

    parser.add_argument(
        "--config",
        "--exp",
        dest="exp",
        type=str,
        required=True,
        help="Path to experiment YAML config file (alias: --exp)",
    )
    parser.add_argument(
        "--backend_path",
        nargs="?",
        default=None,
        help=(
            "Optional backend import path for Megatron or TorchTitan. "
            "If provided, it will be appended to PYTHONPATH dynamically."
        ),
    )
    parser.add_argument(
        "--export_config",
        type=str,
        help="Optional path to export the final merged config to a file.",
    )

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    return parser.parse_known_args() if ignore_unknown_args else (parser.parse_args(), [])


def _parse_kv_overrides(args: list[str]) -> dict:
    """
    Parse CLI arguments of the form:
      --key=value
      --key value
      --flag (boolean True)
    into a nested dictionary structure.

    Supports nested keys using dot notation, e.g., --a.b.c=1.
    """
    overrides = {}
    i = 0
    while i < len(args):
        arg = args[i]
        # Ignore non-option arguments (not starting with "--")
        if not arg.startswith("--"):
            i += 1
            continue

        # Strip the "--" prefix
        key = arg[2:]

        if "=" in key:
            # Format: --key=value
            key, val = key.split("=", 1)
        elif i + 1 < len(args) and not args[i + 1].startswith("--"):
            # Format: --key value
            val = args[i + 1]
            i += 1
        else:
            # Format: --flag (boolean True)
            val = True

        # Try to evaluate the value to correct type (int, float, bool, etc.)
        try:
            val = eval(val, {}, {})
        except Exception:
            pass  # Leave as string if evaluation fails

        # Handle nested keys, e.g., modules.pre_trainer.lr
        d = overrides
        keys = key.split(".")
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val

        i += 1

    return overrides


def _deep_merge_namespace(ns, override_dict):
    """
    Merge override_dict into namespace ns without validation.
    Creates new attributes if they don't exist.
    """
    for k, v in override_dict.items():
        if hasattr(ns, k) and isinstance(getattr(ns, k), SimpleNamespace) and isinstance(v, dict):
            _deep_merge_namespace(getattr(ns, k), v)
        else:
            setattr(ns, k, v)


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    args, unknown_args = _parse_args(extra_args_provider, ignore_unknown_args=True)

    config_parser = PrimusParser()
    primus_config = config_parser.parse(args)

    # Merge CLI overrides directly without validation
    overrides = _parse_kv_overrides(unknown_args)
    pre_trainer_cfg = primus_config.get_module_config("pre_trainer")
    _deep_merge_namespace(pre_trainer_cfg, overrides)

    return primus_config


def load_primus_config(args: argparse.Namespace, overrides: List[str]) -> Tuple[Any, Dict[str, Any]]:
    """
    Build the Primus configuration with optional command-line overrides.

    Args:
        args: Parsed CLI arguments.
        overrides: Key-value pairs from CLI for overriding configs, e.g.:
                   ["training.steps", "1000", "optimizer.lr", "0.001"]

    Returns:
        The merged Primus configuration namespace.
    """
    # 1 Parse the base config from args
    config_parser = PrimusParser()
    primus_config = config_parser.parse(args)

    # 2 Parse overrides from flat list to dict/namespace
    override_dict = _parse_kv_overrides(overrides)

    # 3 Merge all overrides directly into the first trainer module found
    # The trainer will validate parameters itself
    try:
        # Find first trainer module (pre_trainer, sft_trainer, etc.)
        for module_name in primus_config.module_keys:
            module_cfg = primus_config.get_module_config(module_name)
            if hasattr(module_cfg, "framework"):  # It's a trainer module
                _deep_merge_namespace(module_cfg, override_dict)
                break
    except (ValueError, AttributeError):
        # No trainer module found, that's okay (might be benchmark-only config)
        pass

    return primus_config, override_dict


class PrimusParser(object):
    def __init__(self):
        pass

    def parse(self, cli_args: argparse.Namespace) -> PrimusConfig:
        exp_yaml_cfg = cli_args.config
        self.primus_home = Path(os.path.dirname(__file__)).parent.parent.absolute()
        self.parse_exp(exp_yaml_cfg)
        self.parse_meta_info()
        self.parse_platform()
        self.parse_modules()
        return PrimusConfig(cli_args, self.exp)

    def parse_exp(self, config_file: str):
        self.exp = yaml_utils.parse_yaml_to_namespace(config_file)
        self.exp.name = constant_vars.PRIMUS_CONFIG_NAME
        self.exp.config_file = config_file

    def parse_meta_info(self):
        yaml_utils.check_key_in_namespace(self.exp, "work_group")
        yaml_utils.check_key_in_namespace(self.exp, "user_name")
        yaml_utils.check_key_in_namespace(self.exp, "exp_name")
        yaml_utils.check_key_in_namespace(self.exp, "workspace")

    def parse_platform(self):
        # If platform is set in exp config
        if not hasattr(self.exp, "platform"):
            self.exp.platform = SimpleNamespace(
                config="platform_azure.yaml", overrides=SimpleNamespace(master_sink_level="INFO")
            )

        # Load platform config
        config_path = os.path.join(self.primus_home, "configs/platforms", self.exp.platform.config)
        platform_config = yaml_utils.parse_yaml_to_namespace(config_path)
        platform_config.config = self.exp.platform.config

        # Optional overrides
        if yaml_utils.has_key_in_namespace(self.exp.platform, "overrides"):
            yaml_utils.override_namespace(platform_config, self.exp.platform.overrides)

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

        yaml_utils.set_value_by_key(self.exp, "platform", platform_config, allow_override=True)

    def get_model_format(self, framework: str):
        map = {
            "megatron": "megatron",
            "light-megatron": "megatron",
            "torchtitan": "torchtitan",
            "maxtext": "maxtext",
        }
        assert framework in map, f"Invalid module framework: {framework}."
        return map[framework]

    def parse_trainer_module(self, module_name: str, module_def: SimpleNamespace):
        """
        Parse a trainer module that has config/model files to load and merge.

        Args:
            module_name: Name of the module (e.g., "pretrain_stage")
            module_def: Raw module definition from YAML
        """
        # Ensure framework exists
        if not hasattr(module_def, "framework"):
            raise ValueError(f"Module '{module_name}' is missing required 'framework' field")

        framework = module_def.framework

        # If both config and model are missing, skip complex parsing
        has_config = hasattr(module_def, "config")
        has_model = hasattr(module_def, "model")

        if not has_config and not has_model:
            # No config/model files to load, just use the raw definition
            module_def.name = module_name
            yaml_utils.set_value_by_key(self.exp.modules, module_name, module_def, allow_override=True)
            return

        # Both config and model are required if one is present
        if not (has_config and has_model):
            raise ValueError(
                f"Module '{module_name}' must have both 'config' and 'model' fields, "
                f"or neither (for inline configuration)"
            )

        # ---- Load module config ----
        model_format = self.get_model_format(framework)
        module_config_file = os.path.join(
            self.primus_home, "configs/modules", model_format, module_def.config
        )
        module_config = yaml_utils.parse_yaml_to_namespace(module_config_file)
        module_config.name = f"exp.modules.{module_name}.config"
        module_config.framework = framework

        # ---- Load model config ----
        model_config_file = os.path.join(self.primus_home, "configs/models", model_format, module_def.model)
        model_config = yaml_utils.parse_yaml_to_namespace(model_config_file)
        model_config.name = f"exp.modules.{module_name}.model"

        # ---- Merge: config + model ----
        yaml_utils.merge_namespace(module_config, model_config, allow_override=False, excepts=["name"])

        # ---- Apply overrides if present ----
        if hasattr(module_def, "overrides"):
            yaml_utils.override_namespace(module_config, module_def.overrides)

        # ---- Fill in framework defaults ----
        if model_format == "megatron":
            from primus.modules.trainer.megatron.config_loader import (
                apply_megatron_defaults,
            )

            apply_megatron_defaults(module_config)

        # ---- Flatten and save back ----
        module_config.name = module_name
        yaml_utils.set_value_by_key(self.exp.modules, module_name, module_config, allow_override=True)

    def parse_tool_module(self, module_name: str, module_def: SimpleNamespace):
        """
        Parse a tool/benchmark module (no config/model files, just raw params).

        Args:
            module_name: Name of the module (e.g., "gemm_bench")
            module_def: Raw module definition from YAML
        """
        # Tool modules are used as-is, no complex merging
        module_def.name = module_name
        yaml_utils.set_value_by_key(self.exp.modules, module_name, module_def, allow_override=True)

    def parse_modules(self):
        yaml_utils.check_key_in_namespace(self.exp, "modules")

        # Iterate over normalized modules (SimpleNamespace keyed by name)
        for module_name, module_def in vars(self.exp.modules).items():
            if not isinstance(module_def, SimpleNamespace):
                raise TypeError(f"Module '{module_name}' must be a SimpleNamespace, got {type(module_def)}")

            # Dispatch based on 'module' type field
            module_type = getattr(module_def, "module", None)

            if not module_type:
                # Fallback: check if it has 'framework' (trainer) or not (tool)
                if hasattr(module_def, "framework"):
                    self.parse_trainer_module(module_name, module_def)
                else:
                    self.parse_tool_module(module_name, module_def)
            elif module_type in ("pre_trainer", "sft_trainer"):
                self.parse_trainer_module(module_name, module_def)
            elif module_type.startswith("benchmark_"):
                self.parse_tool_module(module_name, module_def)
            else:
                # Unknown module type, treat as tool
                self.parse_tool_module(module_name, module_def)

    def export(self, export_path):
        """
        Export the merged Primus config (exp) to YAML.
        """
        path = Path(export_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = yaml_utils.nested_namespace_to_dict(self.exp)
        yaml_utils.dump_namespace_to_yaml(data, str(path))
        print(f"[PrimusConfig] Exported merged config to {path}")
        return path
