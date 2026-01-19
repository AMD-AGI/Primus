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

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Load Megatron-Bridge default configuration
# ------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_megatron_bridge_defaults() -> Dict[str, Any]:
    """
    Load Megatron-Bridge's default configuration values as a dictionary.

    Note: ConfigContainer requires 8 mandatory keyword-only arguments, but each
    of these config classes can be instantiated with default values using **{}.
    We create minimal default instances for each required field.

    Returns:
        Dictionary of default configuration values from Megatron-Bridge
    """
    try:
        from megatron.bridge.training.config import (
            CheckpointConfig,
            ConfigContainer,
            FinetuningDatasetConfig,
            LoggerConfig,
            OptimizerConfig,
            SchedulerConfig,
            TokenizerConfig,
            TrainingConfig,
        )
        from megatron.bridge.models import GPTModelProvider

        # Create minimal default instances for all required fields
        # Each config class has default values for its fields
        config_container = ConfigContainer(
            train=TrainingConfig(),
            model=GPTModelProvider(),
            optimizer=OptimizerConfig(),
            scheduler=SchedulerConfig(),
            dataset=FinetuningDatasetConfig(),
            logger=LoggerConfig(),
            tokenizer=TokenizerConfig(),
            checkpoint=CheckpointConfig(),
        )

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
        """
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
