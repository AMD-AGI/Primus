###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Argument builder for Megatron-Bridge backend.

Megatron-Bridge uses a recipe-based configuration system built on top of
Megatron-Core. This builder translates Primus configs to Megatron-Bridge
compatible arguments while supporting both traditional args and recipe configs.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Union

from primus.core.config.merge_utils import deep_merge
from primus.core.utils.yaml_utils import (
    dict_to_nested_namespace,
    nested_namespace_to_dict,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# MegatronBridgeArgBuilder: merge Primus → Megatron-Bridge
# ------------------------------------------------------------
class MegatronBridgeArgBuilder:
    """
    A lightweight utility to build final Megatron-Bridge arguments for Primus.

    It merges:
        1. Primus CLI arguments
        2. Primus config arguments
        3. Megatron-Bridge's default values
        4. Recipe-based configurations (if specified)

    Usage:
        builder = MegatronBridgeArgBuilder()
        builder.update(cli_args)
        builder.update(config_args)
        bridge_ns = builder.finalize()

    'bridge_ns' is a SimpleNamespace containing all fields Megatron-Bridge expects.
    """

    def __init__(self):
        # Load Megatron-Bridge recipe configuration
        # self.config = load_recipe_config(module_config.params).to_dict()
        self.config: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Add values to the configuration
    # ------------------------------------------------------------------
    def update(self, values: Union[Mapping[str, Any], SimpleNamespace]) -> "MegatronBridgeArgBuilder":
        """
        Merge a collection of values (e.g., CLI args or config) into the
        current configuration set.

        - Supports both Mapping (e.g., dict) and SimpleNamespace inputs.
        - None values are allowed and will override defaults.
        - Only keys that exist in self.config will be merged (unknown keys are ignored).
        """
        # # Convert SimpleNamespace to dict
        # values_dict = nested_namespace_to_dict(values)

        # # Filter: only keep keys that exist in self.config
        # filtered_values = _filter_existing_keys(self.config, values_dict)

        # log_dict_aligned("Filtered values", filtered_values)

        # # Merge filtered values into the working configuration
        # self.config = deep_merge(self.config, filtered_values)
        # return self
        # Convert SimpleNamespace to dict
        values_dict = nested_namespace_to_dict(values)

        # Directly merge into the working configuration
        self.config = deep_merge(self.config, values_dict)
        return self

    # ------------------------------------------------------------------
    # Produce the final Megatron-Bridge ConfigContainer
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Return a copy of the current configuration as a nested dictionary.

        The configuration already contains:
            - Megatron-Bridge default ConfigContainer values (loaded during __init__)
            - Primus overrides (applied via update() calls)

        This is an intermediate representation before materializing
        the final ConfigContainer dataclass.

        Note: Returns a deep copy to prevent external modifications.
        """
        import copy

        return copy.deepcopy(self.config)

    def to_namespace(self) -> SimpleNamespace:
        """
        Produce the final Megatron-Bridge configuration as a SimpleNamespace.

        This method ensures API consistency with MegatronArgBuilder.to_namespace().
        The namespace contains a nested structure matching TorchTitan's JobConfig.

        Fields not provided by Primus are automatically filled with TorchTitan's defaults.

        Returns:
            SimpleNamespace with nested TorchTitan configuration that can be passed
            to convert back to JobConfig when needed
        """
        merged = self.to_dict()
        return dict_to_nested_namespace(merged)

    # Alias for usage style: builder.finalize()
    finalize = to_namespace
