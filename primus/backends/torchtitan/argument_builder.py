###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan argument builder utilities.

This module is the TorchTitan counterpart of
``primus.backends.megatron.argument_builder``.

It provides a small helper to:
    - load TorchTitan's default JobConfig values
    - apply nested overrides from Primus config or CLI
    - materialize a final ``JobConfig`` dataclass instance
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

from primus.core.config.merge_utils import deep_merge
from primus.core.utils.yaml_utils import (
    dict_to_nested_namespace,
    nested_namespace_to_dict,
)

# -----------------------------------------------------------------------------
# Load TorchTitan's default JobConfig as a nested dict
# -----------------------------------------------------------------------------


def _load_torchtitan_defaults() -> Dict[str, Any]:
    """
    Load TorchTitan's default JobConfig values as a nested dictionary.

    This is analogous to Megatron's ``_load_megatron_defaults`` helper, but
    for TorchTitan's dataclass-based configuration.
    """
    from torchtitan.config.job_config import JobConfig

    return JobConfig().to_dict()


# -----------------------------------------------------------------------------
# TorchTitanJobConfigBuilder
# -----------------------------------------------------------------------------


class TorchTitanJobConfigBuilder:
    """
    A lightweight utility to build final TorchTitan ``JobConfig`` for Primus.

    It merges:
        1. Primus CLI arguments
        2. Primus config arguments
        3. TorchTitan's default JobConfig values

    WITHOUT defining any manual mapping and WITHOUT maintaining version compatibility
    manually — because we rely entirely on TorchTitan's own JobConfig dataclass.

    Usage:
        builder = TorchTitanJobConfigBuilder()
        builder.update(cli_args)
        builder.update(config_args)
        namespace = builder.finalize()

    'namespace' is a SimpleNamespace containing all fields TorchTitan expects.
    """

    def __init__(self) -> None:
        # Load TorchTitan defaults once during initialization
        # and store as the working configuration that will be updated
        self.config: Dict[str, Any] = _load_torchtitan_defaults()

    # ------------------------------------------------------------------
    # Add values to the configuration
    # ------------------------------------------------------------------
    def update(self, values: SimpleNamespace) -> "TorchTitanJobConfigBuilder":
        """
        Merge a SimpleNamespace into the current configuration.

        - Only accepts SimpleNamespace inputs (with nested structure)
        - All parameters are accepted (TorchTitan's JobConfig is flexible)
        - Values are directly merged into the working configuration

        The structure of ``values`` should follow TorchTitan's JobConfig layout,
        with nested SimpleNamespace objects, e.g.:

            SimpleNamespace(
                model=SimpleNamespace(name="llama3", flavor="debugmodel"),
                training=SimpleNamespace(steps=1000),
                parallelism=SimpleNamespace(tensor_parallel_degree=4)
            )
        """
        # Convert SimpleNamespace to dict
        values_dict = nested_namespace_to_dict(values)

        # Directly merge into the working configuration
        self.config = deep_merge(self.config, values_dict)
        return self

    # ------------------------------------------------------------------
    # Produce the final TorchTitan JobConfig
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Return a copy of the current configuration as a nested dictionary.

        The configuration already contains:
            - TorchTitan default JobConfig values (loaded during __init__)
            - Primus overrides (applied via update() calls)

        This is an intermediate representation before materializing
        the final JobConfig dataclass or SimpleNamespace.

        Note: Returns a deep copy to prevent external modifications.
        """
        import copy

        return copy.deepcopy(self.config)

    def to_namespace(self) -> SimpleNamespace:
        """
        Produce the final TorchTitan configuration as a SimpleNamespace.

        This method ensures API consistency with MegatronArgBuilder.to_namespace().
        The namespace contains a nested structure matching TorchTitan's JobConfig.

        Fields not provided by Primus are automatically filled with TorchTitan's defaults.

        Returns:
            SimpleNamespace with nested TorchTitan configuration that can be passed
            to convert back to JobConfig when needed
        """
        merged = self.to_dict()
        return dict_to_nested_namespace(merged)

    # Alias for usage style consistency with MegatronArgBuilder:
    # builder.finalize()
    finalize = to_namespace
