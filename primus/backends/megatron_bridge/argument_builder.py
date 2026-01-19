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
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Union

from primus.core.config.merge_utils import deep_merge
from primus.core.utils.yaml_utils import (
    dict_to_nested_namespace,
    nested_namespace_to_dict,
)
from primus.modules.module_utils import log_dict_aligned

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
            # Check basic type compatibility (allow None)
            if value is not None and base_value is not None:
                base_type = type(base_value)
                value_type = type(value)
                
                # Allow numeric type conversions (int <-> float)
                if (base_type in (int, float) and value_type in (int, float)):
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
            else:
                # Allow None values to override
                filtered[key] = value
    
    return filtered


# ------------------------------------------------------------
# Load Megatron-Bridge default configuration
# ------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_megatron_bridge_defaults() -> Dict[str, Any]:
    """
    Load Megatron-Bridge's default configuration values as a dictionary.

    This function automatically instantiates all required config classes with
    default values using reflection, making it maintainable and extensible.

    Returns:
        Dictionary of default configuration values from Megatron-Bridge
    """
    try:
        from dataclasses import MISSING, fields as dataclass_fields

        from megatron.bridge.training.config import ConfigContainer

        # Special cases: configs that need specific default arguments
        special_defaults = {
            "dataset": ("FinetuningDatasetConfig", {"seq_length": 2048}),
            "model": ("GPTModelProvider", {}),
        }

        # Auto-instantiate all required fields
        kwargs = {}
        for field in dataclass_fields(ConfigContainer):
            # Skip optional fields (those with default or default_factory)
            if field.default is not MISSING or field.default_factory is not MISSING:
                continue

            field_name = field.name

            # Check if this field needs special handling
            if field_name in special_defaults:
                class_name, init_kwargs = special_defaults[field_name]
                # Import the class dynamically
                if class_name == "GPTModelProvider":
                    from megatron.bridge.models import GPTModelProvider

                    kwargs[field_name] = GPTModelProvider(**init_kwargs)
                elif class_name == "FinetuningDatasetConfig":
                    from megatron.bridge.training.config import FinetuningDatasetConfig

                    kwargs[field_name] = FinetuningDatasetConfig(**init_kwargs)
            else:
                # Auto-instantiate using field type annotation
                field_type = field.type
                # Handle Union types by extracting the first non-None type
                if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                    field_type = next(
                        t for t in field_type.__args__ if t is not type(None)
                    )

                # Try to instantiate with no arguments
                try:
                    kwargs[field_name] = field_type()
                except Exception as e:
                    logger.warning(
                        f"Failed to auto-instantiate {field_name} ({field_type}): {e}. "
                        f"Skipping this field."
                    )
                    continue

        # Create ConfigContainer with all required fields
        config_container = ConfigContainer(**kwargs)

        log_dict_aligned("Megatron-Bridge defaults", config_container.to_dict())
        return config_container.to_dict()

    except ImportError as e:
        logger.warning(
            f"Failed to import Megatron-Bridge config classes: {e}. "
            "Returning empty dict. Make sure Megatron-Bridge is installed."
        )
        return {}


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
        # Load Megatron-Bridge defaults once during initialization
        self.config: Dict[str, Any] = _load_megatron_bridge_defaults()

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
