import argparse
from typing import Any, Dict, List, Tuple

from primus.core.launcher.primus_parser import PrimusParser


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
