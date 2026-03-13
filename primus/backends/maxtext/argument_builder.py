###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText Configuration Builder.

- ``MaxTextConfigBuilder``
  Light-weight builder used by ``MaxTextAdapter.convert_config()`` to
  normalise Primus ``module_config.params`` (a ``SimpleNamespace``) into
  the canonical form expected by downstream code.  No heavy dependencies
  (JAX, OmegaConf …) are imported at this stage.

- ``export_params_to_yaml()``
  Writes a flat config dict to a temporary YAML file so that MaxText's
  ``pyconfig.initialize(argv)`` can load it.  Config parsing is entirely
  delegated to MaxText, ensuring version compatibility with both the
  Pydantic-based (Dec) and dict-based (Aug) config systems.

- ``namespace_to_dict()``
  Recursively converts ``SimpleNamespace`` to plain dicts.
"""

from __future__ import annotations

import logging
import os
import tempfile
from types import SimpleNamespace
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ============================================================================
# Lightweight builder (no JAX / OmegaConf at import time)
# ============================================================================


class MaxTextConfigBuilder:
    """
    Builder for MaxText configuration.

    Takes Primus module config parameters and produces a ``SimpleNamespace``
    that represents the canonical set of MaxText parameters.
    """

    def __init__(self):
        self.config = SimpleNamespace()

    def update(self, params: SimpleNamespace):
        """Absorb Primus params (already merged with CLI overrides)."""
        self.config = params

    def finalize(self) -> SimpleNamespace:
        """Return the config namespace for downstream use."""
        return self.config


# ============================================================================
# Helpers
# ============================================================================


def namespace_to_dict(obj: Any) -> Any:
    """Recursively convert ``SimpleNamespace`` (and nested containers) to dict."""
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [namespace_to_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(namespace_to_dict(v) for v in obj)
    return obj


def export_params_to_yaml(params_dict: Dict[str, Any]) -> str:
    """
    Write a flat config dict to a temporary YAML file for ``pyconfig.initialize``.

    Parameters
    ----------
    params_dict : dict
        Flat configuration dictionary produced by
        ``namespace_to_dict(backend_args)``.

    Returns
    -------
    yaml_path : str
        Path to the temporary YAML file.  The **caller** is responsible for
        deleting this file when it is no longer needed.
    """
    import yaml

    fd, yaml_path = tempfile.mkstemp(suffix=".yml", prefix="primus_maxtext_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(params_dict, f, default_flow_style=False, allow_unicode=True)
    return yaml_path
