###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
MaxDiffusion Configuration Builder.

Mirrors ``primus/backends/maxtext/argument_builder.py``:

- ``MaxDiffusionConfigBuilder``
  Light-weight builder used by ``MaxDiffusionAdapter.convert_config()`` to
  normalise Primus ``module_config.params`` (a ``SimpleNamespace``) into the
  form expected by downstream code. No heavy dependencies (JAX, MaxDiffusion)
  are imported here.

- ``export_params_to_yaml()``
  Writes a flat config dict to a temporary YAML file so that MaxDiffusion's
  ``pyconfig.initialize(argv)`` can load it as its config file (argv[1]).

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


class MaxDiffusionConfigBuilder:
    """Builder for MaxDiffusion configuration.

    Takes Primus module config parameters and returns the canonical set of
    MaxDiffusion parameters as a ``SimpleNamespace``.
    """

    def __init__(self):
        self.config = SimpleNamespace()

    def update(self, params: SimpleNamespace):
        """Absorb Primus params (already merged with CLI overrides)."""
        self.config = params

    def finalize(self) -> SimpleNamespace:
        """Return the config namespace for downstream use."""
        return self.config


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
    """Write a config dict to a temporary YAML file for ``pyconfig.initialize``.

    The caller is responsible for deleting the returned file.
    """
    import yaml

    fd, yaml_path = tempfile.mkstemp(suffix=".yml", prefix="primus_maxdiffusion_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(params_dict, f, default_flow_style=False, allow_unicode=True)
    return yaml_path
