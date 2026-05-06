###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Core Primus configuration helpers (YAML → SimpleNamespace).

This module provides a very small helper API used by the new core runtime:

  - load_primus_config: load experiment YAML to a nested SimpleNamespace
  - get_module_config:  fetch a single module config by name

It intentionally does **not** define a PrimusConfig class to avoid
confusion with `primus.core.launcher.config.PrimusConfig`.
"""

from __future__ import annotations

from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from primus.core.launcher.parser import PrimusParser
from primus.core.utils import constant_vars
from primus.core.utils.yaml_utils import (
    dict_to_nested_namespace,
    nested_namespace_to_dict,
)


def _to_plain_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, SimpleNamespace):
        return nested_namespace_to_dict(value)
    return dict(value)


def _normalize_module_for_runtime(module_cfg: SimpleNamespace, module_name: str) -> SimpleNamespace:
    """
    Normalize a legacy module namespace into the runtime shape:
       - Keep a small set of reserved top-level attributes:
           * name, framework, config, model, launcher
           * (and any existing 'params' if present)
       - Move all other public attributes into a `params` dict:
           module_cfg.params[key] = <original value>
    """
    normalized = deepcopy(module_cfg)
    # Ensure each module has a stable `.name` attribute.
    if not getattr(normalized, "name", None):
        setattr(normalized, "name", module_name)

    reserved_keys = {"name", "framework", "config", "model", "params", "launcher"}
    # Start from any existing params dict/namespace if provided.
    existing_params = getattr(normalized, "params", {})
    params = _to_plain_dict(existing_params)

    for key, value in list(vars(normalized).items()):
        # Skip reserved and private attributes.
        if key in reserved_keys or key.startswith("_"):
            continue
        # Move this attribute into params and remove it from the namespace.
        params[key] = value
        delattr(normalized, key)

    # Convert params dict to nested namespace for attribute-style access.
    normalized.params = dict_to_nested_namespace(params)
    # Duplicate `model` under `params.model` for downstream compatibility.
    if hasattr(normalized, "model"):
        normalized.params.model = normalized.model
    return normalized


def load_primus_config(config_path: Path, cli_args: Any | None = None) -> SimpleNamespace:
    """
    Load a Primus experiment YAML file into a lightweight SimpleNamespace,
    using the legacy `PrimusParser` (`PrimusConfig` interface) under the hood.

    Differences from `primus.core.launcher.config.PrimusConfig`:
        - Returns a plain SimpleNamespace instead of PrimusConfig
        - `cfg.modules` is a **list** of module namespaces
          (each module namespace has a `.name` field)
    """
    config_path_str = str(config_path)

    # Build an argparse-like namespace for PrimusParser.
    # PrimusParser.parse() only requires a `.config` attribute.
    if cli_args is None:
        args_for_parser = SimpleNamespace(config=config_path_str)
    elif getattr(cli_args, "config", None) != config_path_str:
        # Avoid mutating the original args object.
        args_for_parser = copy(cli_args)
        setattr(args_for_parser, "config", config_path_str)
    else:
        args_for_parser = cli_args

    # Use legacy PrimusParser to build a PrimusConfig instance.
    legacy_cfg = PrimusParser().parse(args_for_parser)

    # Adapt legacy PrimusConfig → SimpleNamespace for the new runtime.
    cfg = SimpleNamespace()
    cfg.name = constant_vars.PRIMUS_CONFIG_NAME
    cfg.config_file = config_path_str

    # Keep a reference to CLI args for compatibility with utilities
    # such as `set_global_variables`, which expect `cli_args`.
    if cli_args is not None:
        cfg.cli_args = cli_args

    # Copy key properties from legacy PrimusConfig.
    cfg.exp_root_path = legacy_cfg.exp_root_path
    cfg.exp_meta_info = legacy_cfg.exp_meta_info

    # Platform configuration is exposed via `platform_config`.
    platform_config = legacy_cfg.platform_config
    cfg.platform_config = platform_config
    # Also keep `platform` for compatibility with helpers that expect it.
    cfg.platform = platform_config

    # Build modules list from legacy PrimusConfig.module_keys/get_module_config.
    cfg.modules = [
        _normalize_module_for_runtime(legacy_cfg.get_module_config(module_name), module_name)
        for module_name in getattr(legacy_cfg, "module_keys", [])
    ]

    return cfg


def get_module_config(cfg: SimpleNamespace, module_name: str) -> SimpleNamespace | None:
    """
    Fetch a single module config from `cfg.modules` by name.
    """
    modules = getattr(cfg, "modules", None) or []
    for m in modules:
        if getattr(m, "name", None) == module_name:
            return m

    return None


def get_module_names(cfg: SimpleNamespace) -> list[str]:
    """
    Return a list of module names from `cfg.modules`.
    """
    modules = getattr(cfg, "modules", None) or []
    return [getattr(m, "name", None) for m in modules if getattr(m, "name", None)]
