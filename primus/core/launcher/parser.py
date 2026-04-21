from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from primus.core.config.preset_loader import PresetLoader
from primus.core.launcher.config import PrimusConfig
from primus.core.utils import constant_vars, yaml_utils
from primus.core.utils.arg_utils import parse_cli_overrides


def _add_common_train_args(
    parser: argparse.ArgumentParser, include_data_path: bool
) -> argparse.ArgumentParser:
    """
    Register shared train CLI arguments.
    """
    parser.add_argument(
        "--config",
        "--exp",
        dest="config",
        type=str,
        required=True,
        help="Path to experiment YAML config file (alias: --exp)",
    )
    if include_data_path:
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


def add_pretrain_parser(parser: argparse.ArgumentParser):
    return _add_common_train_args(parser, include_data_path=True)


def add_posttrain_parser(parser: argparse.ArgumentParser):
    """
    Post-training (SFT / alignment) workflow parser.

    For now, posttrain shares the same top-level CLI arguments as pretrain
    (config path, data path, optional backend path, export config).
    """
    return add_pretrain_parser(parser)


def _parse_args(extra_args_provider=None, ignore_unknown_args=False) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Primus Arguments", allow_abbrev=False)
    parser = _add_common_train_args(parser, include_data_path=False)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    return parser.parse_known_args() if ignore_unknown_args else (parser.parse_args(), [])


def _parse_kv_overrides(args: list[str]) -> dict:
    """
    Backward-compatible wrapper around the shared CLI override parser.

    Keep this symbol for existing callers/tests in legacy paths.
    """
    return parse_cli_overrides(args, type_mode="legacy")


def _deep_merge_namespace(ns, override_dict):
    """
    Merge overrides into SimpleNamespace via unified yaml merge path.
    """
    yaml_utils.deep_merge_namespace(ns, override_dict)


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


def _allow_unknown_cli_overrides(pre_trainer_cfg: SimpleNamespace) -> bool:
    """
    Decide whether CLI override keys should be merged directly into module config
    even when those keys are not declared in the module YAML preset.
    """
    launcher_cfg = getattr(pre_trainer_cfg, "launcher", None)
    if launcher_cfg is None:
        return False
    return bool(getattr(launcher_cfg, "allow_unknown_cli_overrides", False))


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    args, unknown_args = _parse_args(extra_args_provider, ignore_unknown_args=True)

    config_parser = PrimusParser()
    primus_config = config_parser.parse(args)

    overrides = parse_cli_overrides(unknown_args, type_mode="legacy")
    pre_trainer_cfg = primus_config.get_module_config("pre_trainer")
    if not _allow_unknown_cli_overrides(pre_trainer_cfg):
        _check_keys_exist(pre_trainer_cfg, overrides)
    _deep_merge_namespace(pre_trainer_cfg, overrides)

    return primus_config


def _load_legacy_primus_config(args: argparse.Namespace, overrides: List[str]) -> Tuple[Any, Dict[str, Any]]:
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
    override_ns = parse_cli_overrides(overrides, type_mode="legacy")

    # 3 Apply overrides to pre_trainer module config
    pre_trainer_cfg = primus_config.get_module_config("pre_trainer")
    # _check_keys_exist(pre_trainer_cfg, override_ns)
    # _deep_merge_namespace(pre_trainer_cfg, override_ns)

    if _allow_unknown_cli_overrides(pre_trainer_cfg):
        # Megatron legacy flow should not depend on YAML key presence. This keeps
        # overrides stable after removing redundant/null keys from module presets.
        _deep_merge_namespace(pre_trainer_cfg, override_ns)
        return primus_config, {}
    known_overrides, unknown_overrides = _split_known_unknown(pre_trainer_cfg, override_ns)

    if known_overrides:
        _deep_merge_namespace(pre_trainer_cfg, known_overrides)

    if unknown_overrides:
        print(f"[PrimusConfig] Detected unknown override keys: {list(unknown_overrides.keys())}")

    return primus_config, unknown_overrides


def load_primus_config(args: argparse.Namespace, overrides: List[str]) -> Tuple[Any, Dict[str, Any]]:
    """
    Legacy compatibility API.

    Prefer `primus.core.config.primus_config.load_primus_config` in new code.
    """
    return _load_legacy_primus_config(args, overrides)


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
            "megatron_bridge": "megatron_bridge",
            "torchtitan": "torchtitan",
            "maxtext": "maxtext",
        }
        # If framework not in map, return framework itself as fallback
        return map.get(framework, framework)

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
        for key in ("config", "model"):
            yaml_utils.check_key_in_namespace(module, key)

        # ---- Load module config ----
        model_format = self.get_model_format(framework)

        module_config_dict = PresetLoader.load(module.config, model_format, config_type="modules")
        module_config = yaml_utils.dict_to_nested_namespace(module_config_dict)
        module_config.name = f"exp.modules.{module_name}.config"
        module_config.framework = framework

        # ---- Load model config ----
        model_config_dict = PresetLoader.load(module.model, model_format, config_type="models")
        model_config = yaml_utils.dict_to_nested_namespace(model_config_dict)
        model_config.name = f"exp.modules.{module_name}.model"
        # Only set the top-level `model` field when it does not already exist in the
        # loaded model preset, so that presets can define their own `model` metadata.
        if not yaml_utils.has_key_in_namespace(model_config, "model"):
            model_config.model = module.model

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
