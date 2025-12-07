import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from primus.core.config.preset_loader import PresetLoader
from primus.core.launcher.config import PrimusConfig
from primus.core.utils import constant_vars, yaml_utils


def add_pretrain_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config file (alias: --exp)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path to data directory [default: ./data]",
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
    return parser


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
    for k, v in override_dict.items():
        if hasattr(ns, k) and isinstance(getattr(ns, k), SimpleNamespace) and isinstance(v, dict):
            _deep_merge_namespace(getattr(ns, k), v)
        else:
            setattr(ns, k, v)


def _check_keys_exist(ns: SimpleNamespace, overrides: dict, prefix=""):
    for k, v in overrides.items():
        full_key = f"{prefix}.{k}" if prefix else k
        assert hasattr(ns, k), f"Override key '{full_key}' does not exist in pre_trainer config."
        attr_val = getattr(ns, k)
        if isinstance(v, dict):
            assert isinstance(
                attr_val, SimpleNamespace
            ), f"Override key '{full_key}' expects a namespace/dict but got {type(attr_val)}"
            _check_keys_exist(attr_val, v, prefix=full_key)


def _split_known_unknown(ns: SimpleNamespace, overrides: dict) -> Tuple[dict, dict]:
    """
    Split overrides into two dictionaries:
      - known: keys that exist in the namespace
      - unknown: keys not defined in the namespace
    """
    known, unknown = {}, {}
    for k, v in overrides.items():
        if hasattr(ns, k):
            attr_val = getattr(ns, k)
            if isinstance(v, dict) and isinstance(attr_val, SimpleNamespace):
                sub_known, sub_unknown = _split_known_unknown(attr_val, v)
                if sub_known:
                    known[k] = sub_known
                if sub_unknown:
                    unknown[k] = sub_unknown
            else:
                known[k] = v
        else:
            unknown[k] = v
            # print(f"[PrimusConfig] Unknown key '{k}' delegated to backend.")
    return known, unknown


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    args, unknown_args = _parse_args(extra_args_provider, ignore_unknown_args=True)

    config_parser = PrimusParser()
    primus_config = config_parser.parse(args)

    overrides = _parse_kv_overrides(unknown_args)
    pre_trainer_cfg = primus_config.get_module_config("pre_trainer")
    _check_keys_exist(pre_trainer_cfg, overrides)
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
    override_ns = _parse_kv_overrides(overrides)

    # 3 Apply overrides to pre_trainer module config
    pre_trainer_cfg = primus_config.get_module_config("pre_trainer")
    # _check_keys_exist(pre_trainer_cfg, override_ns)
    # _deep_merge_namespace(pre_trainer_cfg, override_ns)

    # return primus_config
    known_overrides, unknown_overrides = _split_known_unknown(pre_trainer_cfg, override_ns)

    if known_overrides:
        _deep_merge_namespace(pre_trainer_cfg, known_overrides)

    if unknown_overrides:
        print(f"[PrimusConfig] Detected unknown override keys: {list(unknown_overrides.keys())}")

    return primus_config, unknown_overrides


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

    def parse_trainer_module(self, module_name: str):
        module = yaml_utils.get_value_by_key(self.exp.modules, module_name)
        module.name = f"exp.modules.{module_name}"

        # Ensure framework exists
        yaml_utils.check_key_in_namespace(module, "framework")
        framework = module.framework

        # If both config and model are missing, skip
        has_config = yaml_utils.has_key_in_namespace(module, "config")
        has_model = yaml_utils.has_key_in_namespace(module, "model")
        if not has_config and not has_model:
            return

        # If we only have a model but no config, assume this module has already
        # been flattened (e.g., from a previously exported config) and skip
        # re-processing. This allows PrimusParser.export → parse cycles.
        if not has_config and has_model:
            return

        # Validate required keys
        # for key in ("config", "model"):
        #     yaml_utils.check_key_in_namespace(module, key)

        # ---- Load module config ----
        model_format = self.get_model_format(framework)

        module_config_dict = PresetLoader.load(module.config, model_format, config_type="modules")
        module_config = yaml_utils.dict_to_nested_namespace(module_config_dict)
        module_config.name = f"exp.modules.{module_name}.config"
        module_config.framework = framework
        # module_config.model = module.model

        # ---- Load model config ----
        model_config_dict = PresetLoader.load(module.model, model_format, config_type="models")
        model_config = yaml_utils.dict_to_nested_namespace(model_config_dict)
        model_config.name = f"exp.modules.{module_name}.model"

        # Avoid 'model' key conflicts when merging module + model presets:
        # - Keep module_config.model as the user-specified model identifier
        # - Treat detailed model metadata from the model preset as 'model_info'
        # if hasattr(model_config, "model"):
        #     setattr(model_config, "model_info", getattr(model_config, "model"))
        #     delattr(model_config, "model")

        # ---- Merge: config + model ----
        yaml_utils.merge_namespace(module_config, model_config, allow_override=False, excepts=["name"])

        # ---- Apply overrides if present ----
        if yaml_utils.has_key_in_namespace(module, "overrides"):
            yaml_utils.override_namespace(module_config, module.overrides)

        # ---- Flatten and save back ----
        module_config.name = module_name
        yaml_utils.set_value_by_key(self.exp.modules, module_name, module_config, allow_override=True)

    def parse_modules(self):
        yaml_utils.check_key_in_namespace(self.exp, "modules")
        for module_name in vars(self.exp.modules):
            if "trainer" in module_name:
                self.parse_trainer_module(module_name)
            else:
                raise ValueError(f"Unsupported module: {module_name}")

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
