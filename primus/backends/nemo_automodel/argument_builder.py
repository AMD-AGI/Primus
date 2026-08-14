###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
NeMo AutoModel configuration builder.

This is the NeMo AutoModel counterpart of ``MaxTextConfigBuilder`` /
``TorchTitanJobConfigBuilder``. AutoModel (like MaxText) is an external trainer
that owns its own configuration system: a recipe is driven by a single YAML that
``nemo_automodel.components.config._arg_parser.parse_args_and_load_config`` loads
into a ``ConfigNode``. So there is no JobConfig-style dataclass of defaults to
merge against — the Primus ``module_config.params`` (already the AutoModel config
schema: ``model:``/``fsdp:``/``optim:``/``step_scheduler:``/...) *is* the config.

This builder therefore just normalises ``params`` (a ``SimpleNamespace``) and the
trainer materialises it into an AutoModel ``ConfigNode`` via AutoModel's own
loader (see ``nemo_automodel_pretrain_trainer.py``), keeping us version-agnostic.
"""

from __future__ import annotations

import logging
import os
import tempfile
from types import SimpleNamespace
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Top-level keys injected by Primus (module_base.yaml / runtime) that are not
# part of the AutoModel config schema. Stripped before handing the dict to
# AutoModel. NOTE: this is TOP-LEVEL only — nested AutoModel keys such as
# ``model.stage`` (the high/low-noise expert window) must be preserved.
#
# This is a denylist: a new top-level key added to module_base.yaml would
# otherwise leak into the AutoModel ConfigNode. ``test_module_base_keys_are_all_stripped``
# (tests/unit_tests/backends/nemo_automodel) guards against that drift — if it
# fails, add the new key here (or confirm it belongs in the AutoModel recipe).
PRIMUS_ONLY_TOP_KEYS = frozenset(
    {
        "trainable",
        "sink_level",
        "file_sink_level",
        "stderr_sink_level",
        "framework",
        "config",
        "name",
        "stage",  # Primus trainer-stage selector; AutoModel uses model.stage
    }
)


class NemoAutomodelConfigBuilder:
    """Absorb Primus ``module_config.params`` and return it for downstream use."""

    def __init__(self) -> None:
        self.config: SimpleNamespace = SimpleNamespace()

    def update(self, params: SimpleNamespace) -> "NemoAutomodelConfigBuilder":
        """Take params (already merged with model preset + CLI overrides)."""
        self.config = params
        return self

    def finalize(self) -> SimpleNamespace:
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


def strip_primus_keys(params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Drop Primus-only top-level keys so AutoModel sees a clean recipe config."""
    return {k: v for k, v in params_dict.items() if k not in PRIMUS_ONLY_TOP_KEYS}


_PUBLISHED_PARAMS: Dict[str, Any] = {}


def publish_params(params_dict: Dict[str, Any]) -> None:
    """Make the resolved YAML params readable by the optional install hooks.

    ``_install_optional_hooks`` runs before the recipe (and therefore its config object)
    exists, and every hook is a bare ``install()`` with no config argument — so a hook that
    wants to be driven by a YAML key rather than an env var has nothing to read. The
    trainer publishes the merged params here first; :func:`get_param` is the read side.
    Keys read this way must be ones AutoModel ignores: the recipe pulls ``model.*`` fields
    it knows by name and never splats the block into a constructor, so an extra flag there
    is inert to it.
    """
    global _PUBLISHED_PARAMS
    _PUBLISHED_PARAMS = params_dict


def get_param(dotted_key: str, default: Any = None) -> Any:
    """Read ``a.b.c`` from the published params; ``default`` if absent or never published.

    Absent-means-default is deliberate: a hook must behave identically when it is imported
    outside a run (unit tests, ``python -c`` checks), where nothing was ever published.
    """
    node: Any = _PUBLISHED_PARAMS
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def export_params_to_yaml(params_dict: Dict[str, Any]) -> str:
    """
    Write the (cleaned) config dict to a temporary YAML so AutoModel's own
    ``parse_args_and_load_config`` can load it — exactly mirroring the MaxText
    backend. The caller is responsible for deleting the file.
    """
    import yaml

    fd, yaml_path = tempfile.mkstemp(suffix=".yaml", prefix="primus_automodel_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(params_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return yaml_path
