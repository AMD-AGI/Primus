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
  - get_module_map:     get the mapping of module_name -> module_namespace

It intentionally does **not** define a PrimusConfig class to avoid
confusion with `primus.core.launcher.config.PrimusConfig`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

from primus.core.utils import constant_vars, yaml_utils


def load_primus_config(config_path: Path, cli_args: Any | None = None) -> SimpleNamespace:
    """
    Load a Primus experiment YAML file into a nested SimpleNamespace.

    This helper is intentionally minimal and leaves most semantics
    (e.g., how modules are structured) to the YAML schema itself.
    """
    exp = yaml_utils.parse_yaml_to_namespace(str(config_path))
    exp.name = constant_vars.PRIMUS_CONFIG_NAME
    exp.config_file = str(config_path)

    # Attach optional helpers to keep compatibility with utilities
    # like `set_global_variables`, which expect `cli_args` and
    # `platform_config` attributes.
    if cli_args is not None:
        setattr(exp, "cli_args", cli_args)

    if hasattr(exp, "platform") and not hasattr(exp, "platform_config"):
        setattr(exp, "platform_config", exp.platform)

    return exp


def get_module_config(cfg: SimpleNamespace, module_name: str) -> SimpleNamespace:
    """
    Fetch a single module config from `cfg.modules` by name.
    """
    modules = getattr(cfg, "modules", None)
    if modules is None or not hasattr(modules, module_name):
        raise ValueError(
            f"Primus config ({getattr(cfg, 'config_file', '')}) has no module named {module_name}"
        )
    return getattr(modules, module_name)


def get_module_map(cfg: SimpleNamespace) -> Dict[str, Any]:
    """
    Return a plain dict mapping module_name -> module_namespace.
    """
    modules = getattr(cfg, "modules", None)
    if modules is None:
        return {}
    return vars(modules)
