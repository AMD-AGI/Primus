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

from primus.backends.megatron_bridge.config_utils import load_recipe_config
from primus.core.config.merge_utils import deep_merge
from primus.core.utils.yaml_utils import (
    dict_to_nested_namespace,
    nested_namespace_to_dict,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def _filter_existing_keys(base: Dict[str, Any], updates: Dict[str, Any], path: str = "") -> Dict[str, Any]:
    """
    Filter updates dict to only include keys that exist in base dict with type checking.

    Recursively filters nested dictionaries to ensure:
    1. Only existing keys are updated
    2. Type compatibility is maintained (warns on mismatch)

    Args:
        base: Base dictionary (defines which keys are allowed and their types)
        updates: Updates dictionary (may contain keys not in base or wrong types)
        path: Current path in nested structure (for logging)

    Returns:
        Filtered updates dict with only keys that exist in base and have matching types
    """
    filtered = {}

    for key, value in updates.items():
        current_path = f"{path}.{key}" if path else key

        if key not in base:
            # Skip keys that don't exist in base
            logger.debug(f"Ignoring unknown config key: {current_path}")
            continue

        base_value = base[key]

        # Type checking
        base_is_dict = isinstance(base_value, dict)
        value_is_dict = isinstance(value, dict)

        if base_is_dict and value_is_dict:
            # Both are dicts, recursively filter
            filtered_nested = _filter_existing_keys(base_value, value, current_path)
            if filtered_nested:  # Only include if there are valid nested keys
                filtered[key] = filtered_nested
        elif base_is_dict and not value_is_dict:
            # Type mismatch: base is dict but value is not
            logger.warning(
                f"Type mismatch for '{current_path}': "
                f"expected dict (base type), got {type(value).__name__}. "
                f"Skipping update."
            )
            continue
        elif not base_is_dict and value_is_dict:
            # Type mismatch: base is not dict but value is
            logger.warning(
                f"Type mismatch for '{current_path}': "
                f"expected {type(base_value).__name__} (base type), got dict. "
                f"Skipping update."
            )
            continue
        else:
            # Both are non-dict values
            # Skip None values to preserve recipe defaults
            if value is None:
                logger.debug(f"Skipping None value for '{current_path}', keeping recipe default")
                continue
            
            # Skip if base is None (no default to compare against)
            if base_value is None:
                # User provided a value but base has None - allow it
                filtered[key] = value
                continue
            
            # Type compatibility check (both non-None)
            base_type = type(base_value)
            value_type = type(value)

            # Allow numeric type conversions (int <-> float)
            if base_type in (int, float) and value_type in (int, float):
                filtered[key] = value
            elif base_type == value_type:
                # Same type, allow update
                filtered[key] = value
            else:
                # Type mismatch
                logger.warning(
                    f"Type mismatch for '{current_path}': "
                    f"expected {base_type.__name__} (base: {base_value}), "
                    f"got {value_type.__name__} (update: {value}). "
                    f"Skipping update."
                )
                continue

    return filtered


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

    def __init__(self, module_config: SimpleNamespace):
        # Load Megatron-Bridge recipe configuration
        self.config = load_recipe_config(module_config.params).to_dict()

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
        # Convert SimpleNamespace to dict
        values_dict = nested_namespace_to_dict(values)

        # Filter: only keep keys that exist in self.config
        filtered_values = _filter_existing_keys(self.config, values_dict)

        # Merge filtered values into the working configuration
        self.config = deep_merge(self.config, filtered_values)
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

    # def to_config_container(self) -> Any:
    #     """
    #     Produce the final Megatron-Bridge ConfigContainer with user overrides applied.

    #     This method:
    #     1. Starts with the recipe ConfigContainer
    #     2. Applies user overrides from self.config (merged via update())
    #     3. Returns updated ConfigContainer

    #     Returns:
    #         ConfigContainer with recipe defaults + user overrides
    #     """
    #     from megatron.bridge.training.config import ConfigContainer
    #     from megatron.bridge.training.utils.config_utils import InstantiationMode

    #     # Get the merged config dict
    #     merged_dict = self.to_dict()

    #     # Add _target_ field for from_dict()
    #     merged_dict["_target_"] = "megatron.bridge.training.config.ConfigContainer"

    #     # Reconstruct ConfigContainer from merged dict
    #     try:
    #         config_container = ConfigContainer.from_dict(merged_dict, mode=InstantiationMode.LENIENT)
    #     except Exception as e:
    #         logger.warning(f"Failed to create ConfigContainer with LENIENT mode: {e}")
    #         logger.warning("Falling back to direct dict merge on original ConfigContainer")

    #         # Fallback: use OmegaConf to merge
    #         try:
    #             from omegaconf import OmegaConf
    #             from megatron.bridge.training.utils.omegaconf_utils import (
    #                 apply_overrides,
    #                 create_omegaconf_dict_config,
    #             )

    #             # Start with the original recipe ConfigContainer
    #             config_container = self.config_container

    #             # Convert to OmegaConf
    #             omega_cfg, excluded = create_omegaconf_dict_config(config_container)

    #             # Apply overrides
    #             apply_overrides(config_container, merged_dict, excluded)
    #         except Exception as e2:
    #             logger.error(f"Failed to merge config with OmegaConf: {e2}")
    #             # Last resort: return original recipe config
    #             logger.warning("Returning original recipe ConfigContainer without user overrides")
    #             config_container = self.config_container

    #     return config_container

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
