###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText Configuration Builder.

Converts Primus configuration to MaxText PyConfig format.
"""

from types import SimpleNamespace
from typing import Any, Dict


class MaxTextConfigBuilder:
    """
    Builder for MaxText configuration.

    This takes Primus module config parameters and produces a configuration
    object suitable for MaxText training.
    """

    def __init__(self):
        self.config = SimpleNamespace()

    def update(self, params: Dict[str, Any]):
        """
        Update configuration with parameters from Primus config.

        Args:
            params: Dictionary of configuration parameters
        """
        for key, value in params.items():
            setattr(self.config, key, value)

    def finalize(self) -> SimpleNamespace:
        """
        Finalize and return the MaxText configuration.

        Returns:
            SimpleNamespace with MaxText configuration
        """
        return self.config
